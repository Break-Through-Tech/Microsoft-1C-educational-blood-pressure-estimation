import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, ScrollRestoration, useLocation } from "react-router";
import { Menu, Monitor, Moon, Sun } from "lucide-react";
import { DISCLAIMER } from "@/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/radix";
import { useHealth } from "@/lib/queries";
import { useTheme, type ThemeChoice } from "@/lib/theme";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/predict", label: "Predict" },
  { to: "/compare", label: "Compare" },
  { to: "/explain", label: "Explain" },
  { to: "/try", label: "Try your own" },
  { to: "/about", label: "About" },
];

/** Always visible, never dismissible. Text matches the API's `disclaimer`. */
export function DisclaimerBanner() {
  const health = useHealth();
  return (
    <div className="border-b border-rule bg-subtle px-4 py-2 text-center text-xs text-ink sm:text-sm" role="note" aria-label="Disclaimer">
      {health.data?.disclaimer ?? DISCLAIMER}
    </div>
  );
}

export function ModeBadge() {
  const health = useHealth();
  if (health.data?.mode !== "demo") return null;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="rounded-full">
          <Badge variant="caution">Demo data</Badge>
        </button>
      </TooltipTrigger>
      <TooltipContent>
        Placeholder models and synthetic signals. Run the evaluation script to switch to real results.
      </TooltipContent>
    </Tooltip>
  );
}

const THEME_NEXT: Record<ThemeChoice, ThemeChoice> = { system: "light", light: "dark", dark: "system" };
const THEME_LABEL: Record<ThemeChoice, string> = {
  system: "Theme: follows your system",
  light: "Theme: light",
  dark: "Theme: dark",
};

export function ThemeToggle() {
  const { choice, setChoice } = useTheme();
  const Icon = choice === "system" ? Monitor : choice === "light" ? Sun : Moon;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon" onClick={() => setChoice(THEME_NEXT[choice])} aria-label={`${THEME_LABEL[choice]}. Click to change.`}>
          <Icon />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{THEME_LABEL[choice]}</TooltipContent>
    </Tooltip>
  );
}

function Wordmark() {
  return (
    <Link to="/" className="flex items-center gap-2 rounded-md text-lg font-bold tracking-[-0.02em]">
      <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
        <rect width="32" height="32" rx="8" className="fill-led" />
        <path d="M4 18h6l2.5-7 4 13 3-9 1.5 3H28" fill="none" className="stroke-led-ink" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      CufflessAI
    </Link>
  );
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    "rounded-md px-1 py-1 text-sm transition-colors",
    isActive ? "font-semibold text-ink underline decoration-led decoration-2 underline-offset-[22px]" : "text-muted-foreground hover:text-ink",
  );

function ServerProblem() {
  const health = useHealth();
  if (!health.isError) return null;
  return (
    <div className="bg-caution-soft px-4 py-2 text-center text-sm" role="alert">
      {health.error instanceof Error ? health.error.message : "Can't reach the server."}{" "}
      <button type="button" className="font-semibold underline underline-offset-2" onClick={() => health.refetch()}>
        Try again
      </button>
    </div>
  );
}

export function AppShell() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);

  return (
    <div className="flex min-h-dvh flex-col overflow-x-clip">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-surface focus:px-4 focus:py-2">
        Skip to content
      </a>
      <DisclaimerBanner />
      <ServerProblem />
      <header className="sticky top-0 z-40 border-b border-rule bg-paper">
        <div className="mx-auto flex h-16 max-w-[1120px] items-center gap-4 px-6">
          <Wordmark />
          <ModeBadge />
          <nav aria-label="Main" className="ml-auto hidden items-center gap-6 md:flex">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} className={navClass}>
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center md:ml-2">
            <ThemeToggle />
            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open menu">
                  <Menu />
                </Button>
              </SheetTrigger>
              <SheetContent>
                <SheetTitle className="text-lg font-semibold">Menu</SheetTitle>
                <SheetDescription className="sr-only">Site pages</SheetDescription>
                <nav aria-label="Main" className="flex flex-col gap-1">
                  {[{ to: "/", label: "Home" }, ...NAV].map((n) => (
                    <NavLink
                      key={n.to}
                      to={n.to}
                      end
                      className={({ isActive }) =>
                        cn("rounded-md px-4 py-4 text-base", isActive ? "bg-subtle font-semibold" : "text-muted-foreground")
                      }
                    >
                      {n.label}
                    </NavLink>
                  ))}
                </nav>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </header>

      <main id="main" className="mx-auto w-full max-w-[1120px] flex-1 px-6 pt-14 pb-28 sm:pt-20">
        <Outlet />
      </main>

      <footer className="border-t border-rule">
        <div className="mx-auto flex max-w-[1120px] flex-col gap-2 px-6 py-8 text-xs text-muted-foreground sm:flex-row sm:justify-between">
          <p>A Break Through Tech AI Studio project, hosted by Microsoft.</p>
          <p>Nothing you upload is stored.</p>
        </div>
      </footer>
      <ScrollRestoration />
    </div>
  );
}
