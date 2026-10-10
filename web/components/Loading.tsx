// A gentle loading state that says what is happening (Claude-written "why"s take a few seconds the first time).
export default function Loading({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center gap-4 py-20 text-center text-muted">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-accent" />
      <p className="text-sm">{message}</p>
    </div>
  );
}
