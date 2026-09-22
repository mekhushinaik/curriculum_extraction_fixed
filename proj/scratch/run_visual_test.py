from visual.pipeline import process_pdf_visuals

res = process_pdf_visuals("jess103.pdf", "Social Science", "10")
print("Finished processing visual sidecar!")

for page in res["pages"]:
    figs = page.get("figures", [])
    if figs:
        print(f"\n=== Page {page['page_number']} ({len(figs)} figures) ===")
        for f in figs:
            print(f"  -> {f['figure_id']} | Type: {f['figure_type']} | Desc: {f['description']}")
