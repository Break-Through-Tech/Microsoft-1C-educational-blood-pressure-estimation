import { keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { api, type PredictRequest } from "@/api";

// The runtime is loaded once per server start, so these change rarely: cache generously.
const STATIC = { staleTime: 5 * 60_000 } as const;

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: api.health, ...STATIC });
export const useModels = () => useQuery({ queryKey: ["models"], queryFn: api.models, ...STATIC });
export const useSamples = () => useQuery({ queryKey: ["samples"], queryFn: api.samples, ...STATIC });
export const useMetrics = () => useQuery({ queryKey: ["metrics"], queryFn: api.metrics, ...STATIC });

export const useSample = (id: number | null) =>
  useQuery({
    queryKey: ["sample", id],
    queryFn: () => api.sample(id!),
    enabled: id !== null,
    placeholderData: keepPreviousData,
    ...STATIC,
  });

/** Explanations are deterministic per (model, sample), so they are queries, not mutations. */
export const useExplain = (model: string | null, sampleId: number | null) =>
  useQuery({
    queryKey: ["explain", model, sampleId],
    queryFn: () => api.explain({ model: model!, sample_id: sampleId! }),
    enabled: model !== null && sampleId !== null,
    placeholderData: keepPreviousData,
    ...STATIC,
  });

/** Predictions are mutations: a user's signal must never sit in a cache keyed by its content. */
export const usePredict = () => useMutation({ mutationFn: (body: PredictRequest) => api.predict(body) });
