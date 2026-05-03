# Pipeline architecture

A visual overview of how a TCGA slide travels from the GDC servers to
a CIN signature prediction. Rendered diagrams below are mermaid; GitHub
renders them inline.

## End-to-end flow

```mermaid
flowchart TD
    subgraph Ingest["Stream ingest (per cohort, parallel workers)"]
        A1[GDC manifest<br/>tcga_<cohort>.tsv] --> A2{shard<br/>round-robin}
        A2 --> W0[worker 0<br/>GPU 0]
        A2 --> W1[worker 1<br/>GPU 1]
        A2 --> W2[worker 2<br/>GPU 2]
        A2 --> W3[worker 3<br/>GPU 3]
        W0 --> D[gdc-client<br/>download .svs]
        W1 --> D
        W2 --> D
        W3 --> D
        D --> T[tile_wsi.py<br/>20× tile + Otsu mask<br/>+ Macenko stain]
        T --> E[embed_tiles.py<br/>frozen UNI ViT-L/16<br/>fp16 inference]
        E --> H[(.h5 embeddings<br/>~300 MB / cohort)]
        E -.->|delete| X1[discard .svs]
        T -.->|delete| X2[discard tile cache]
    end
    H --> M[build_train_manifest.py<br/>min 20 tiles, dedupe]
    M --> Train

    subgraph Train["Training (sweep + final retrain)"]
        S1[5-fold cancer-type<br/>stratified CV] --> S2{per-fold<br/>active mask}
        S2 --> S3[gated ABMIL head<br/>multi-task Huber<br/>AdamW + early stop]
        S3 --> S4[(fold checkpoints<br/>cv_summary.json)]
    end

    Train --> Eval

    subgraph Eval["Evaluation + interpretability"]
        V1[per-fold rho<br/>+ MAE + R²]
        V2[robustness:<br/>BH-FDR + bootstrap]
        V3[per-cohort breakdown]
        V4[attention rollout<br/>+ prototype tiles]
    end

    Eval --> Out[(figures/*.pdf<br/>dashboard_7cohort.html<br/>predictions.tsv)]
```

## Stream-mode invariant

The pipeline never holds the full pan-cancer dataset on disk. At any
moment, peak disk footprint is bounded by:

```
peak ≈ N_workers × max(slide_size) + total_embeddings + transient_tiles
     ≈   4       ×   ~2 GB         + ~10 GB total    + ~few hundred MB
     ≈ ~20 GB
```

Final state for a complete pan-TCGA run (33 cohorts, ~30k slides) is
about 10 GB of embeddings. Raw slides are deleted as soon as their
`.h5` is written.

## GPU pinning

```mermaid
flowchart LR
    L[parallel_ingest_8cohort.sh]
    L -->|CUDA_VISIBLE_DEVICES=0,1| GA["Group A: ESCA → PAAD<br/>→ COAD → KIRP"]
    L -->|CUDA_VISIBLE_DEVICES=2,3| GB["Group B: CESC → THYM<br/>→ PCPG → STAD"]
    GA --> SP1[stream_ingest_parallel.py<br/>n-workers=2]
    GB --> SP2[stream_ingest_parallel.py<br/>n-workers=2]
    SP1 -->|worker 0| P0[physical GPU 0]
    SP1 -->|worker 1| P1[physical GPU 1]
    SP2 -->|worker 0| P2[physical GPU 2]
    SP2 -->|worker 1| P3[physical GPU 3]
```

The `stream_ingest_parallel.py` script reads the parent's
`CUDA_VISIBLE_DEVICES` if set, splits it on commas, and assigns
`available_gpus[k % len(available_gpus)]` to worker k. This is what
makes two-cohort-parallel ingest actually use four physical GPUs
instead of collapsing onto the first two.

## Active signature mask

```mermaid
flowchart LR
    L[CX labels<br/>per fold train split] --> V[var > 1e-5 ?]
    V -->|yes| A[active]
    V -->|no| I[inactive]
    A --> Loss[contributes to<br/>MultiTaskHuber]
    A --> Rho[contributes to<br/>per-fold ρ]
    I --> Skip[masked out:<br/>loss = 0, ρ = NaN]
```

Pan-TCGA, this masks out CX7 and CX15 (zero in 100% of patients),
CX14 (99.9% zero), and CX13 (93% zero) in any fold whose training or
validation split is constant. The masked dimensions still flow through
the model but contribute nothing to gradient or metric.

## Stage 5 finalization (auto-triggered)

```mermaid
flowchart TD
    F0[finalize_15cohort.sh<br/>polls every 60 s] --> F1{all 8 .done?}
    F1 -->|no| F0
    F1 -->|yes| F2[build_train_manifest.py<br/>15 cohorts]
    F2 --> F3[train_cv.py<br/>h512 d0.3 lr1e-4<br/>cuda:0]
    F3 --> F4[render figure]
    F4 --> F5[git add + commit + push]
```

This is what makes the overnight run unattended: the launcher exits
after kicking off ingest, the auto-finalizer takes over once the eight
cohort markers all exist, and the headline 15-cohort figure lands on
`origin/main` without manual intervention.
