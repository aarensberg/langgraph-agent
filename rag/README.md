# `rag/` — source documents for the RAG tool

The `search_school_documents` tool indexes the PDF files in this folder (Chroma +
local FastEmbed embeddings). The actual files are the student's **personal
official documents**, so they are gitignored (`rag/*.pdf`) and are **not** part
of the submission, per the assignment spec ("do not include large/personal data
files").

To run the agent locally, place these four PDFs here:

| File | What it is |
|---|---|
| `enrollment_certificate.pdf` | Official proof of enrollment |
| `transcript_year-1.pdf` | Yearly transcript — Bachelor year 1 |
| `transcript_semester-1_year-2.pdf` | Semestrial transcript — year 2, S1 |
| `transcript_semester-2_year-2.pdf` | Semestrial transcript — year 2, S2 |

The transcripts are the **official historical record** (French /20 scale, letter
grades, ECTS, PASS/FAIL per teaching unit) — distinct from the live API grades,
which are on a /100 scale. On first run the documents are embedded once and the
vector store is persisted to `.chroma/` (also gitignored); later runs just load
it.

Any set of PDFs works — drop your own files here and the tool will index them.
