import { Link, isRouteErrorResponse, useRouteError } from "react-router";
import { Button } from "@/components/ui/button";

export default function NotFoundPage() {
  return (
    <div className="py-16">
      <h1 className="text-2xl">This page doesn't exist</h1>
      <p className="measure mt-4 text-muted-foreground">The link may be out of date. Pick a page from the menu, or start from the home page.</p>
      <Button asChild className="mt-8">
        <Link to="/">Go to the home page</Link>
      </Button>
    </div>
  );
}

/** Route-level crash: keep the shell, explain, offer a way out. */
export function RouteError() {
  const error = useRouteError();
  if (isRouteErrorResponse(error) && error.status === 404) return <NotFoundPage />;
  return (
    <div className="py-16">
      <h1 className="text-2xl">This page stopped working</h1>
      <p className="measure mt-4 text-muted-foreground">
        Something in the app broke while showing this page. Reload to try again. If it keeps happening, check the browser
        console for details.
      </p>
      <div className="mt-8 flex gap-4">
        <Button onClick={() => window.location.reload()}>Reload page</Button>
        <Button asChild variant="outline">
          <Link to="/">Go to the home page</Link>
        </Button>
      </div>
    </div>
  );
}
