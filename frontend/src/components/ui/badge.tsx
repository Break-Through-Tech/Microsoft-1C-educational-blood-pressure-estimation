import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-[3px] px-1.5 py-px text-xs whitespace-nowrap",
  {
    variants: {
      variant: {
        led: "bg-led-soft text-led",
        outline: "border border-rule-soft text-muted-foreground",
        caution: "bg-caution-soft text-caution",
      },
    },
    defaultVariants: { variant: "outline" },
  },
);

export function Badge({ className, variant, ...props }: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
