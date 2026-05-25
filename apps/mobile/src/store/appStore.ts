import { create } from "zustand";

import {
  createGame,
  createHistoryEntryAfterMove,
  createInitialHistoryEntry,
  HistoryEntry,
  isLegalFen,
  PromotionPiece,
  tryCreateGame,
  setActiveColorInFen,
  setCastlingRightsInFen,
  tryMove,
  type CastlingRights,
} from "@/lib/chess/game";
import type { Color } from "chess.js";
import { buildFen, detectionsFromSquares } from "@/lib/chess/detections";
import { runVisionPipeline } from "@/lib/pipeline";
import { createImagePreview } from "@/lib/storage/imagePreview";
import { createStubDetection } from "@/lib/storage/loadBoardSnapshot";
import {
  deleteBoardSnapshot,
  listBoardSnapshots,
} from "@/lib/storage/boardSnapshots";
import { persistBoardSnapshot } from "@/lib/storage/persistBoard";
import {
  AnalysisResult,
  AppPhase,
  DetectionResult,
  PipelineStep,
  PipelineStepState,
} from "@/types";
import type { SavedBoardSnapshot } from "@/types/boardSnapshot";

const INITIAL_STEPS: PipelineStepState[] = [
  { id: "upload", label: "Uploading image", status: "pending" },
  { id: "detect", label: "Detecting & warping board", status: "pending" },
  { id: "classify", label: "Classifying 64 squares", status: "pending" },
  { id: "validate", label: "Validating board matrix & FEN", status: "pending" },
  { id: "analyze", label: "Running Stockfish", status: "pending" },
];

interface AppStore {
  phase: AppPhase;
  error: string | null;
  fileName: string | null;
  detection: DetectionResult | null;
  analysis: AnalysisResult | null;
  analysisLoading: boolean;
  showEngineArrows: boolean;
  engineMultiPv: 1 | 2 | 3;
  initialBoardFen: string | null;
  boardReady: boolean;
  fen: string;
  orientation: "white" | "black";
  pipelineSteps: PipelineStepState[];
  history: HistoryEntry[];
  currentMoveIndex: number;
  savedBoards: SavedBoardSnapshot[];
  currentSnapshotId: string | null;

  upload: (file: File) => Promise<void>;
  /** Build FEN from last detection squares and enable interactive board if legal. */
  confirmDetectedPosition: () => boolean;
  setFen: (fen: string) => void;
  flipBoard: () => void;
  reset: () => void;
  resetBoard: () => void;
  makeMove: (from: string, to: string, promotion?: PromotionPiece) => boolean;
  nextMove: () => void;
  previousMove: () => void;
  goToMove: (index: number) => void;
  goToStart: () => void;
  goToEnd: () => void;
  setActiveColor: (turn: Color) => void;
  setCastlingRights: (rights: CastlingRights) => void;
  hydrateSavedBoards: () => void;
  loadSavedBoard: (id: string) => void;
  deleteSavedBoard: (id: string) => void;
  setAnalysis: (analysis: AnalysisResult | null) => void;
  setAnalysisLoading: (loading: boolean) => void;
  setShowEngineArrows: (show: boolean) => void;
  setEngineMultiPv: (count: 1 | 2 | 3) => void;
  startAnalysisBoard: (fen: string, title?: string) => void;
}

function updateStep(
  steps: PipelineStepState[],
  active: PipelineStep,
): PipelineStepState[] {
  const order: PipelineStep[] = ["upload", "detect", "classify", "validate", "analyze"];
  const activeIdx = order.indexOf(active);
  return steps.map((step) => {
    const idx = order.indexOf(step.id);
    if (idx < activeIdx) return { ...step, status: "done" };
    if (idx === activeIdx) return { ...step, status: "active" };
    return { ...step, status: "pending" };
  });
}

function completeSteps(steps: PipelineStepState[]): PipelineStepState[] {
  return steps.map((s) => ({ ...s, status: "done" }));
}

function phaseForStep(step: PipelineStep): AppPhase {
  switch (step) {
    case "upload":
      return "uploading";
    case "detect":
      return "detecting";
    case "classify":
      return "classifying";
    case "validate":
      return "validating";
    case "analyze":
      return "analyzing";
    default:
      return "detecting";
  }
}

function emptyHistoryState() {
  return {
    history: [] as HistoryEntry[],
    currentMoveIndex: 0,
  };
}

function initHistory(fen: string) {
  const entry = createInitialHistoryEntry(fen);
  return {
    history: [entry],
    currentMoveIndex: 0,
    fen,
  };
}

function snapshotFen(state: AppStore): string | null {
  if (state.history.length > 0) {
    return state.history[state.currentMoveIndex]?.fen ?? state.history[0]?.fen ?? null;
  }
  return state.fen || null;
}

function autoSaveBoard(
  get: () => AppStore,
  set: (partial: Partial<AppStore>) => void,
  overrides?: { fen?: string; imagePreview?: string; title?: string; newSnapshot?: boolean },
) {
  const state = get();
  const fen = overrides?.fen ?? snapshotFen(state);
  if (!state.boardReady || !fen) return;

  const id = overrides?.newSnapshot ? null : state.currentSnapshotId;
  const { snapshot, boards } = persistBoardSnapshot({
    id,
    fen,
    orientation: state.orientation,
    title: overrides?.title ?? state.fileName ?? "Saved board",
    imagePreview: overrides?.imagePreview,
  });

  set({ savedBoards: boards, currentSnapshotId: snapshot.id });
}

export const useAppStore = create<AppStore>((set, get) => ({
  phase: "idle",
  error: null,
  fileName: null,
  detection: null,
  analysis: null,
  analysisLoading: false,
  showEngineArrows: true,
  engineMultiPv: 3,
  initialBoardFen: null,
  boardReady: false,
  fen: "",
  orientation: "white",
  pipelineSteps: INITIAL_STEPS,
  savedBoards: [],
  currentSnapshotId: null,
  ...emptyHistoryState(),

  hydrateSavedBoards: () => {
    set({ savedBoards: listBoardSnapshots() });
  },

  loadSavedBoard: (id) => {
    const snapshot = listBoardSnapshots().find((board) => board.id === id);
    if (!snapshot) return;

    try {
      createGame(snapshot.fen);
    } catch {
      set({ error: "Saved board has an invalid FEN" });
      return;
    }

    set({
      phase: "ready",
      error: null,
      fileName: snapshot.title,
      detection: createStubDetection(snapshot),
      analysis: null,
      analysisLoading: false,
      boardReady: true,
      initialBoardFen: snapshot.fen,
      orientation: snapshot.orientation,
      currentSnapshotId: snapshot.id,
      pipelineSteps: completeSteps(INITIAL_STEPS),
      ...initHistory(snapshot.fen),
    });
  },

  deleteSavedBoard: (id) => {
    const boards = deleteBoardSnapshot(id);
    const currentSnapshotId = get().currentSnapshotId === id ? null : get().currentSnapshotId;
    set({ savedBoards: boards, currentSnapshotId });
  },

  upload: async (file) => {
    set({
      phase: "uploading",
      error: null,
      fileName: file.name,
      detection: null,
      analysis: null,
      analysisLoading: false,
      initialBoardFen: null,
      boardReady: false,
      fen: "",
      currentSnapshotId: null,
      ...emptyHistoryState(),
      pipelineSteps: INITIAL_STEPS.map((s, i) => ({
        ...s,
        status: i === 0 ? "active" : "pending",
      })),
    });

    try {
      const pipelineResult = await runVisionPipeline(file, (step) => {
        set((state) => ({
          pipelineSteps: updateStep(state.pipelineSteps, step),
          phase: phaseForStep(step),
        }));
      });

      const detection = pipelineResult.detection;

      const interactiveFen = detection.interactiveFen;
      const startFen = interactiveFen ?? "";
      const canPlay = Boolean(detection.boardReady && startFen && isLegalFen(startFen));
      const imagePreview = await createImagePreview(file);

      set({
        phase: "ready",
        detection: canPlay
          ? detection
          : { ...detection, boardReady: false, interactiveFen: null },
        analysis: null,
        analysisLoading: false,
        boardReady: canPlay,
        initialBoardFen: canPlay ? startFen : null,
        ...(canPlay
          ? initHistory(startFen)
          : { fen: detection.fen ?? "", ...emptyHistoryState() }),
        pipelineSteps: completeSteps(get().pipelineSteps),
      });

      if (canPlay) {
        autoSaveBoard(get, set, {
          fen: startFen,
          imagePreview,
          title: file.name,
          newSnapshot: true,
        });
      }
    } catch (err) {
      set({
        phase: "error",
        error: err instanceof Error ? err.message : "Something went wrong processing your image.",
      });
    }
  },

  confirmDetectedPosition: () => {
    const { detection } = get();
    if (!detection?.squares?.length) {
      set({ error: "No detected squares to confirm." });
      return false;
    }
    try {
      const dets = detectionsFromSquares(detection.squares);
      const fen = buildFen(dets, {
        orientation: detection.orientation === "black" ? "black" : "white",
      });
      createGame(fen);
      const historyState = initHistory(fen);
      set({
        error: null,
        ...historyState,
        boardReady: true,
        phase: "ready",
        initialBoardFen: fen,
        detection: {
          ...detection,
          interactiveFen: fen,
          fen: fen.split(" ")[0],
          boardReady: true,
          fenValid: true,
        },
      });
      autoSaveBoard(get, set, { fen, newSnapshot: true });
      return true;
    } catch {
      set({
        error:
          "Could not build a legal position. Try starting position or scan a clearer photo.",
      });
      return false;
    }
  },

  setFen: (fen) => {
    try {
      createGame(fen);
      const historyState = initHistory(fen);
      set({ error: null, ...historyState, boardReady: true, phase: "ready" });
      if (get().boardReady || get().phase === "ready") {
        set({ initialBoardFen: fen });
      }
      autoSaveBoard(get, set, { fen, newSnapshot: true });
    } catch {
      set({ error: "Invalid FEN string" });
    }
  },

  flipBoard: () => {
    set((s) => ({
      orientation: s.orientation === "white" ? "black" : "white",
    }));
    autoSaveBoard(get, set);
  },

  reset: () =>
    set({
      phase: "idle",
      error: null,
      fileName: null,
      detection: null,
      analysis: null,
      analysisLoading: false,
      initialBoardFen: null,
      boardReady: false,
      fen: "",
      orientation: "white",
      pipelineSteps: INITIAL_STEPS,
      savedBoards: get().savedBoards,
      currentSnapshotId: null,
      ...emptyHistoryState(),
    }),

  resetBoard: () => {
    const { initialBoardFen, boardReady } = get();
    if (!boardReady || !initialBoardFen) return;
    set(initHistory(initialBoardFen));
  },

  goToMove: (index) => {
    const { history } = get();
    if (index < 0 || index >= history.length) return;
    set({
      currentMoveIndex: index,
      fen: history[index].fen,
    });
  },

  nextMove: () => {
    const { currentMoveIndex, history } = get();
    if (currentMoveIndex < history.length - 1) {
      get().goToMove(currentMoveIndex + 1);
    }
  },

  previousMove: () => {
    const { currentMoveIndex } = get();
    if (currentMoveIndex > 0) {
      get().goToMove(currentMoveIndex - 1);
    }
  },

  goToStart: () => get().goToMove(0),

  goToEnd: () => {
    const { history } = get();
    if (history.length > 0) {
      get().goToMove(history.length - 1);
    }
  },

  makeMove: (from, to, promotion) => {
    const { history, currentMoveIndex, boardReady } = get();
    if (!boardReady || history.length === 0) return false;

    const current = history[currentMoveIndex];
    if (!current) return false;

    const result = tryMove(current.fen, from, to, promotion);
    if (!result.ok) return false;

    const entry = createHistoryEntryAfterMove(result.fen, result.move);
    const newHistory = [...history.slice(0, currentMoveIndex + 1), entry];

    set({
      history: newHistory,
      currentMoveIndex: newHistory.length - 1,
      fen: entry.fen,
    });
    return true;
  },

  setActiveColor: (turn) => {
    const { history, boardReady } = get();
    if (!boardReady || history.length !== 1) return;

    const current = history[0];
    if (!current) return;

    try {
      const currentGame = tryCreateGame(current.fen);
      if (!currentGame || currentGame.turn() === turn) return;

      const nextFen = setActiveColorInFen(current.fen, turn);
      const entry = createInitialHistoryEntry(nextFen);

      set({
        history: [entry],
        currentMoveIndex: 0,
        fen: entry.fen,
        initialBoardFen: entry.fen,
        error: null,
      });
      autoSaveBoard(get, set, { fen: entry.fen });
    } catch {
      set({ error: "Could not set side to move for this position" });
    }
  },

  setCastlingRights: (rights) => {
    const { history, boardReady } = get();
    if (!boardReady || history.length !== 1) return;

    const current = history[0];
    if (!current) return;

    try {
      const nextFen = setCastlingRightsInFen(current.fen, rights);
      const entry = createInitialHistoryEntry(nextFen);

      set({
        history: [entry],
        currentMoveIndex: 0,
        fen: entry.fen,
        initialBoardFen: entry.fen,
        error: null,
      });
      autoSaveBoard(get, set, { fen: entry.fen });
    } catch {
      set({ error: "Invalid castling rights for this position" });
    }
  },

  setAnalysis: (analysis) => set({ analysis }),

  setAnalysisLoading: (analysisLoading) => set({ analysisLoading }),

  setShowEngineArrows: (showEngineArrows) => set({ showEngineArrows }),

  setEngineMultiPv: (engineMultiPv) => set({ engineMultiPv }),

  startAnalysisBoard: (fen, title = "Board") => {
    try {
      createGame(fen);
      const now = new Date().toISOString();
      const stub = createStubDetection({
        id: `board-${Date.now()}`,
        fen,
        orientation: "white",
        title,
        createdAt: now,
        updatedAt: now,
        sideToMove: "w",
        castlingRights: {
          whiteKingside: true,
          whiteQueenside: true,
          blackKingside: true,
          blackQueenside: true,
        },
      });

      set({
        phase: "ready",
        error: null,
        fileName: title,
        detection: stub,
        analysis: null,
        analysisLoading: false,
        boardReady: true,
        initialBoardFen: fen,
        orientation: "white",
        currentSnapshotId: null,
        pipelineSteps: completeSteps(INITIAL_STEPS),
        ...initHistory(fen),
      });
    } catch {
      set({ error: "Invalid FEN string", phase: "error" });
    }
  },
}));

export function selectCurrentHistoryEntry(state: AppStore): HistoryEntry | null {
  return state.history[state.currentMoveIndex] ?? null;
}

export function selectIsAtLatestMove(state: AppStore): boolean {
  return state.history.length > 0 && state.currentMoveIndex === state.history.length - 1;
}

export function selectCanEditPosition(state: AppStore): boolean {
  return state.boardReady && state.history.length === 1;
}

/** @deprecated use selectCanEditPosition */
export function selectCanChangeTurn(state: AppStore): boolean {
  return selectCanEditPosition(state);
}
