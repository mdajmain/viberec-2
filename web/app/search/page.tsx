import { Suspense } from "react";
import Loading from "@/components/Loading";
import SearchPage from "./SearchPage";

export default function Page() {
  return (
    <Suspense fallback={<Loading message="Loading…" />}>
      <SearchPage />
    </Suspense>
  );
}
