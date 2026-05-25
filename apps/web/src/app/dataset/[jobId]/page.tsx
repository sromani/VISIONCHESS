import { DatasetReviewPanel } from "@/components/dataset/DatasetReviewPanel";

type PageProps = {
  params: Promise<{ jobId: string }>;
};

export default async function DatasetReviewPage({ params }: PageProps) {
  const { jobId } = await params;
  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <DatasetReviewPanel jobId={jobId} />
    </main>
  );
}
