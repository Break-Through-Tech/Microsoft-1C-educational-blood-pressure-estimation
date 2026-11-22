import { StrictMode, type ComponentType } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError } from "@/api";
import { AppShell, DisclaimerBanner } from "@/components/AppShell";
import { LinesSkeleton } from "@/components/States";
import { TooltipProvider } from "@/components/ui/radix";
import { ThemeProvider } from "@/lib/theme";
import HomePage from "@/pages/Home";
import NotFoundPage, { RouteError } from "@/pages/NotFound";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // don't hammer the server on 4xx (bad id, unknown model); retry once on network/5xx
      retry: (count, err) => count < 1 && !(err instanceof ApiError && err.status >= 400 && err.status < 500),
      refetchOnWindowFocus: false,
    },
  },
});

function PageLoading() {
  return (
    <div className="min-h-dvh">
      <DisclaimerBanner />
      <div className="mx-auto max-w-[1280px] px-6 pt-28">
        <LinesSkeleton lines={4} />
      </div>
    </div>
  );
}

// pages with charts are split out so the home page loads without Recharts
const page = (load: () => Promise<{ default: ComponentType }>) => async () => ({ Component: (await load()).default });

const router = createBrowserRouter([
  {
    element: <AppShell />,
    errorElement: <RouteError />,
    // shown for a moment while a lazily loaded page arrives on first visit
    hydrateFallbackElement: <PageLoading />,
    children: [
      {
        errorElement: <RouteError />,
        children: [
          { path: "/", element: <HomePage /> },
          { path: "/predict", lazy: page(() => import("@/pages/Predict")) },
          { path: "/compare", lazy: page(() => import("@/pages/Compare")) },
          { path: "/explain", lazy: page(() => import("@/pages/Explain")) },
          { path: "/try", lazy: page(() => import("@/pages/TryYourOwn")) },
          { path: "/about", lazy: page(() => import("@/pages/About")) },
          { path: "*", element: <NotFoundPage /> },
        ],
      },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider delayDuration={200}>
          <RouterProvider router={router} />
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
);
