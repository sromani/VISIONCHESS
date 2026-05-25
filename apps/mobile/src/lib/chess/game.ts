import { Chess, type Color, type Move, type Square } from "chess.js";
import type { CustomSquareStyles } from "react-chessboard/dist/chessboard/types";

export type BoardPieceCode =
  | "wP"
  | "wB"
  | "wN"
  | "wR"
  | "wQ"
  | "wK"
  | "bP"
  | "bB"
  | "bN"
  | "bR"
  | "bQ"
  | "bK";

export type PromotionPiece = "q" | "r" | "b" | "n";

export interface LastMove {
  from: Square;
  to: Square;
}

export interface HistoryMove {
  from: Square;
  to: Square;
  san: string;
  color: Color;
  piece: string;
  captured?: string;
  promotion?: string;
}

export interface HistoryEntry {
  move: HistoryMove | null;
  fen: string;
  pgn: string;
  san: string | null;
  lastMove: LastMove | null;
}

export interface GameStatus {
  fen: string;
  turn: Color;
  turnLabel: string;
  inCheck: boolean;
  isCheckmate: boolean;
  isStalemate: boolean;
  isDraw: boolean;
  isGameOver: boolean;
  kingInCheckSquare: Square | null;
}

export interface CastlingRights {
  whiteKingside: boolean;
  whiteQueenside: boolean;
  blackKingside: boolean;
  blackQueenside: boolean;
}

export const EMPTY_CASTLING_RIGHTS: CastlingRights = {
  whiteKingside: false,
  whiteQueenside: false,
  blackKingside: false,
  blackQueenside: false,
};

const SELECTED_STYLE = { backgroundColor: "rgba(20, 85, 30, 0.55)" };
const LAST_MOVE_STYLE = { backgroundColor: "rgba(155, 199, 0, 0.41)" };
const CHECK_STYLE = { backgroundColor: "rgba(255, 0, 0, 0.55)" };

function moveTargetStyle(move: Move): Record<string, string> {
  return {
    background: move.isCapture()
      ? "radial-gradient(circle, rgba(0,0,0,.08) 82%, rgba(220, 45, 45, 0.65) 82%)"
      : "radial-gradient(circle, rgba(0,0,0,.08) 82%, rgba(20, 85, 30, 0.55) 82%)",
  };
}

/** Load FEN without throwing (use in UI / previews). */
export function tryCreateGame(fen: string): Chess | null {
  if (!fen?.trim()) return null;
  try {
    return new Chess(fen);
  } catch {
    return null;
  }
}

export function isLegalFen(fen: string): boolean {
  return tryCreateGame(fen) != null;
}

export function createGame(fen: string): Chess {
  const game = tryCreateGame(fen);
  if (!game) {
    throw new Error(`Invalid FEN: ${fen.trim().split(/\s+/)[0] ?? fen}`);
  }
  return game;
}

export function getActiveColor(fen: string): Color {
  return tryCreateGame(fen)?.turn() ?? "w";
}

export function parseCastlingRights(fen: string): CastlingRights {
  const parts = fen.trim().split(/\s+/);
  const castle = parts[2] ?? "-";
  if (castle === "-") return { ...EMPTY_CASTLING_RIGHTS };

  return {
    whiteKingside: castle.includes("K"),
    whiteQueenside: castle.includes("Q"),
    blackKingside: castle.includes("k"),
    blackQueenside: castle.includes("q"),
  };
}

export function castlingRightsToFenField(rights: CastlingRights): string {
  let field = "";
  if (rights.whiteKingside) field += "K";
  if (rights.whiteQueenside) field += "Q";
  if (rights.blackKingside) field += "k";
  if (rights.blackQueenside) field += "q";
  return field || "-";
}

/** Max castling rights allowed by current king/rook placement (chess.js board). */
export function inferCastlingRights(fen: string): CastlingRights {
  const placement = fen.trim().split(/\s+/)[0];
  const game = tryCreateGame(`${placement} w - - 0 1`);
  if (!game) return { ...EMPTY_CASTLING_RIGHTS };

  const wKing = game.get("e1");
  const wRookH = game.get("h1");
  const wRookA = game.get("a1");
  const bKing = game.get("e8");
  const bRookH = game.get("h8");
  const bRookA = game.get("a8");

  return {
    whiteKingside:
      wKing?.type === "k" && wKing.color === "w" && wRookH?.type === "r" && wRookH.color === "w",
    whiteQueenside:
      wKing?.type === "k" && wKing.color === "w" && wRookA?.type === "r" && wRookA.color === "w",
    blackKingside:
      bKing?.type === "k" && bKing.color === "b" && bRookH?.type === "r" && bRookH.color === "b",
    blackQueenside:
      bKing?.type === "k" && bKing.color === "b" && bRookA?.type === "r" && bRookA.color === "b",
  };
}

function ensureFullFenParts(parts: string[]): string[] {
  const next = [...parts];
  while (next.length < 6) {
    if (next.length === 2) next.push("-");
    else if (next.length === 3) next.push("-");
    else if (next.length === 4) next.push("0");
    else next.push("1");
  }
  return next;
}

export function patchFen(
  fen: string,
  patch: { turn?: Color; castling?: CastlingRights },
): string {
  const parts = ensureFullFenParts(fen.trim().split(/\s+/));
  if (patch.turn) parts[1] = patch.turn;
  if (patch.castling) parts[2] = castlingRightsToFenField(patch.castling);
  const game = tryCreateGame(parts.join(" "));
  if (!game) throw new Error("Invalid FEN");
  return game.fen();
}

/** Replace the active-color field in FEN and re-validate via chess.js. */
export function setActiveColorInFen(fen: string, turn: Color): string {
  return patchFen(fen, { turn });
}

export function setCastlingRightsInFen(fen: string, castling: CastlingRights): string {
  return patchFen(fen, { castling });
}

export function createInitialHistoryEntry(fen: string): HistoryEntry {
  const game = tryCreateGame(fen);
  if (!game) throw new Error("Invalid FEN");
  return {
    move: null,
    fen: game.fen(),
    pgn: game.pgn(),
    san: null,
    lastMove: null,
  };
}

function moveToHistoryMove(move: Move): HistoryMove {
  return {
    from: move.from,
    to: move.to,
    san: move.san,
    color: move.color,
    piece: move.piece,
    captured: move.captured,
    promotion: move.promotion,
  };
}

export function createHistoryEntryAfterMove(fen: string, move: Move): HistoryEntry {
  const game = tryCreateGame(fen);
  if (!game) throw new Error("Invalid FEN");
  return {
    move: moveToHistoryMove(move),
    fen: game.fen(),
    pgn: game.pgn(),
    san: move.san,
    lastMove: { from: move.from, to: move.to },
  };
}

/** True when chess.js reports at least one legal move from this square. */
export function isSquareSelectable(game: Chess, square: Square): boolean {
  return game.moves({ square, verbose: true }).length > 0;
}

export function getLegalMovesForSquare(game: Chess, square: Square): Move[] {
  return game.moves({ square, verbose: true });
}

export function findLegalMove(game: Chess, from: Square, to: Square): Move | undefined {
  return getLegalMovesForSquare(game, from).find((move) => move.to === to);
}

/** Promotion options for from→to, derived from chess.js verbose moves. */
export function getPromotionMoves(game: Chess, from: Square, to: Square): Move[] {
  return getLegalMovesForSquare(game, from).filter(
    (move) => move.to === to && move.isPromotion(),
  );
}

export function getKingSquare(game: Chess, color: Color = game.turn()): Square | null {
  const squares = game.findPiece({ type: "k", color });
  return squares[0] ?? null;
}

export function buildSquareStyles(
  game: Chess,
  selectedSquare: string | null,
  lastMove: LastMove | null,
): CustomSquareStyles {
  const styles: CustomSquareStyles = {};

  if (lastMove) {
    styles[lastMove.from] = LAST_MOVE_STYLE;
    styles[lastMove.to] = LAST_MOVE_STYLE;
  }

  if (game.inCheck()) {
    const kingSquare = getKingSquare(game);
    if (kingSquare) {
      styles[kingSquare] = {
        ...styles[kingSquare],
        ...CHECK_STYLE,
      };
    }
  }

  if (!selectedSquare) return styles;

  styles[selectedSquare as Square] = {
    ...styles[selectedSquare as Square],
    ...SELECTED_STYLE,
  };

  for (const move of getLegalMovesForSquare(game, selectedSquare as Square)) {
    styles[move.to] = {
      ...styles[move.to],
      ...moveTargetStyle(move),
    };
  }

  return styles;
}

export function getGameStatus(fen: string): GameStatus | null {
  const game = tryCreateGame(fen);
  if (!game) return null;
  const turn = game.turn();

  return {
    fen: game.fen(),
    turn,
    turnLabel: turn === "w" ? "White" : "Black",
    inCheck: game.inCheck(),
    isCheckmate: game.isCheckmate(),
    isStalemate: game.isStalemate(),
    isDraw: game.isDraw(),
    isGameOver: game.isGameOver(),
    kingInCheckSquare: game.inCheck() ? getKingSquare(game, turn) : null,
  };
}

export function buildPgn(initialFen: string, moves: string[]): string {
  const game = createGame(initialFen);
  for (const san of moves) {
    game.move(san);
  }
  return game.pgn();
}

export function tryMove(
  fen: string,
  from: string,
  to: string,
  promotion?: PromotionPiece,
): { ok: true; fen: string; san: string; lastMove: LastMove; move: Move } | { ok: false } {
  try {
    const game = createGame(fen);
    const legal = findLegalMove(game, from as Square, to as Square);
    if (!legal) return { ok: false };

    const moveInput = promotion
      ? { from, to, promotion }
      : legal.isPromotion()
        ? null
        : { from, to };

    if (moveInput === null) return { ok: false };

    const result = game.move(moveInput);
    if (!result) return { ok: false };

    return {
      ok: true,
      fen: game.fen(),
      san: result.san,
      lastMove: { from: result.from, to: result.to },
      move: result,
    };
  } catch {
    return { ok: false };
  }
}

export function promotionFromBoardPiece(piece: BoardPieceCode): PromotionPiece {
  return piece[1].toLowerCase() as PromotionPiece;
}
