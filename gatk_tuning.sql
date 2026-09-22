-- Step 1 of the GATK tuning loop: score tables, provenance, per-config averages.
-- Run in the Supabase SQL editor. Safe to re-run.

create extension if not exists pgcrypto;

create table if not exists public.gatk_configs (
  id            uuid primary key default gen_random_uuid(),
  experiment    text not null,
  tool_name     text not null default 'gatk',
  gatk_updates  jsonb not null default '{}'::jsonb,
  gatk_config   jsonb not null default '{}'::jsonb,
  created_at    timestamptz not null default now()
);

create table if not exists public.gatk_evaluations (
  id                     uuid primary key default gen_random_uuid(),
  config_id              uuid not null references public.gatk_configs(id) on delete cascade,
  folder                 text,
  folder_path            text,
  region                 text,
  variant_count          integer,
  scorer                 text not null default 'AdvancedV2',
  scoring_version        text not null default 'v2',
  scoring_status         text,
  ok                     boolean,
  error                  text,
  would_record           boolean,
  score                  double precision,
  combined_final         double precision,
  advanced_score         double precision,
  core                   double precision,
  germline               double precision,
  fp_per_target          double precision,
  snp_fp_per_target      double precision,
  snp_final              double precision,
  indel_final            double precision,
  weighted_f1            double precision,
  f1_snp                 double precision,
  precision_snp          double precision,
  recall_snp             double precision,
  tp_snp                 double precision,
  fp_snp                 double precision,
  fn_snp                 double precision,
  truth_total_snp        double precision,
  query_total_snp        double precision,
  target_total_snp       double precision,
  frac_na_snp            double precision,
  query_unk_snp          double precision,
  f1_indel               double precision,
  precision_indel        double precision,
  recall_indel           double precision,
  tp_indel               double precision,
  fp_indel               double precision,
  fn_indel               double precision,
  truth_total_indel      double precision,
  query_total_indel      double precision,
  target_total_indel     double precision,
  frac_na_indel          double precision,
  query_unk_indel        double precision,
  region_fp_snp          double precision,
  region_fp_indel        double precision,
  region_fp_total        double precision,
  overcall_penalty       double precision,
  titv_query_snp         double precision,
  titv_truth_snp         double precision,
  hethom_query_snp       double precision,
  hethom_truth_snp       double precision,
  hethom_query_indel     double precision,
  hethom_truth_indel     double precision,
  weights                jsonb,
  difficulty_class_counts jsonb,
  metrics                jsonb not null default '{}'::jsonb,
  created_at             timestamptz not null default now()
);

-- Provenance for the agent + Optuna (one config = one trial).
alter table public.gatk_configs add column if not exists study_name text;
alter table public.gatk_configs add column if not exists search_category text;
alter table public.gatk_configs add column if not exists suggested_by text;
alter table public.gatk_configs add column if not exists hypothesis text;
alter table public.gatk_configs add column if not exists parent_config_id uuid;
alter table public.gatk_configs add column if not exists optuna_trial_number integer;
alter table public.gatk_configs add column if not exists status text;

create index if not exists gatk_evaluations_config_id_idx
  on public.gatk_evaluations (config_id);
create index if not exists gatk_evaluations_combined_final_idx
  on public.gatk_evaluations (combined_final desc nulls last);
create index if not exists gatk_configs_experiment_idx
  on public.gatk_configs (experiment);
create index if not exists gatk_configs_category_idx
  on public.gatk_configs (search_category);
create index if not exists gatk_configs_study_idx
  on public.gatk_configs (study_name);

-- One row per config: the number Optuna should maximize.
create or replace view public.gatk_config_scores as
select
  c.id as config_id,
  c.experiment,
  c.tool_name,
  c.study_name,
  c.search_category,
  c.suggested_by,
  c.hypothesis,
  c.parent_config_id,
  c.optuna_trial_number,
  c.status,
  c.gatk_updates,
  c.gatk_config,
  c.created_at,
  count(e.id) as n_folders,
  count(e.id) filter (where e.ok and e.combined_final is not null) as n_scored,
  avg(e.combined_final) filter (where e.ok and e.combined_final is not null)
    as avg_combined_final,
  avg(e.core) filter (where e.ok) as avg_core,
  avg(e.germline) filter (where e.ok) as avg_germline,
  avg(e.fp_per_target) filter (where e.ok) as avg_fp_per_target,
  avg(e.f1_snp) filter (where e.ok) as avg_f1_snp,
  avg(e.f1_indel) filter (where e.ok) as avg_f1_indel,
  avg(e.region_fp_total) filter (where e.ok) as avg_region_fp_total
from public.gatk_configs c
left join public.gatk_evaluations e on e.config_id = c.id
group by c.id;

grant select on public.gatk_config_scores to anon, authenticated, service_role;
