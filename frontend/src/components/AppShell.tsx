import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, ScrollRestoration, useLocation } from "react-router";
import { Menu, Monitor, Moon, Sun } from "lucide-react";
import { DISCLAIMER } from "@/api";
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
    <div className="border-b border-rule-soft px-4 py-1.5 text-center text-xs text-muted-foreground" role="note" aria-label="Disclaimer">
      {health.data?.disclaimer ?? DISCLAIMER}
    </div>
  );
}

/** Thin full-width strip above everything while the server runs on placeholder models. */
export function DemoBanner() {
  const health = useHealth();
  if (health.data?.mode !== "demo") return null;
  return (
    <div className="bg-caution-soft px-4 py-1 text-center text-xs text-caution" role="note" aria-label="Demo mode">
      Demo data — synthetic signals, not real recordings.
    </div>
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

/**
 * One real beat: onset to next onset in app sample id 3 (a held-out UCI window, cleaned), with its
 * dicrotic dip and rebound. Traced from src.preprocess.detect_beats, inverted for SVG y.
 */
const BEAT_PATH =
  "M0 16.8L0.9 16.6 1.8 15.8 2.8 14.3 3.7 12.3 4.6 9.8 5.5 7.2 6.4 4.9 7.3 3.1 8.3 2.1 9.2 2.1 10.1 3.1 11 4.8 11.9 7.1 12.9 9.7 13.8 12.2 14.7 14.3 15.6 15.9 16.5 16.7 17.4 16.9 18.4 16.6 19.3 16 20.2 15.2 21.1 14.6 22 14.3 23 14.3 23.9 14.8 24.8 15.6 25.7 16.5 26.6 17.4 27.5 17.9";

function Wordmark() {
  return (
    <Link to="/" className="flex items-baseline gap-2 rounded-[3px] font-display text-[1.375rem] leading-none font-semibold tracking-[-0.01em]">
      <svg viewBox="-1 0 30 20" className="h-[0.9em] w-auto translate-y-[0.08em] overflow-visible" aria-hidden>
        <path d={BEAT_PATH} fill="none" className="stroke-led" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      CufflessAI
    </Link>
  );
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    "rounded-md px-1 py-1 text-sm transition-colors",
    isActive ? "font-medium text-ink underline decoration-led decoration-2 underline-offset-[19px]" : "text-muted-foreground hover:text-ink",
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
      <DemoBanner />
      <DisclaimerBanner />
      <ServerProblem />
      <header className="sticky top-0 z-40 border-b border-rule-soft bg-paper">
        <div className="mx-auto flex h-14 max-w-[1280px] items-center gap-4 px-6">
          <Wordmark />
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

      <main id="main" className="mx-auto w-full max-w-[1280px] flex-1 px-6 pt-8 pb-24 sm:pt-10">
        <Outlet />
      </main>

      <footer className="border-t border-rule-soft">
        <div className="mx-auto flex max-w-[1280px] flex-col gap-2 px-6 py-8 text-xs text-muted-foreground sm:flex-row sm:justify-between">
          <p>A Break Through Tech AI Studio project, hosted by Microsoft.</p>
          <p>Nothing you upload is stored.</p>
        </div>
      </footer>
      <ScrollRestoration />
    </div>
  );
}
