import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path


def load_visual_data(json_path: str) -> dict | None:
    """
    Attempts to locate and load matching visual_analysis_*.json file.
    """
    path_obj = Path(json_path)
    parent_dir = path_obj.parent
    
    # Try finding matching visual analysis file
    # e.g., output_social-science.json -> visual_analysis_social-science.json
    filename = path_obj.name
    if filename.startswith("output_"):
        visual_filename = filename.replace("output_", "visual_analysis_")
        visual_path = parent_dir / visual_filename
        if visual_path.exists():
            try:
                with open(visual_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    # Fallback search for any visual_analysis_*.json in the directory
    visual_candidates = list(parent_dir.glob("visual_analysis_*.json"))
    if visual_candidates:
        try:
            with open(visual_candidates[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return None


def index_figures(visual_data: dict | None) -> dict:
    """
    Indexes figures by related_subtopic, related_topic, and related_chapter,
    filtering out non-figure entries like 'Do you know?' boxes and page banners.
    """
    indexed = {
        "by_subtopic": {},
        "by_topic": {},
        "by_chapter": {},
        "all_figures": []
    }

    if not visual_data or "pages" not in visual_data:
        return indexed

    try:
        from image_extractor import is_valid_figure
    except ImportError:
        is_valid_figure = None

    for page in visual_data.get("pages", []):
        page_num = page.get("page_number")
        for fig in page.get("figures", []):
            if is_valid_figure and not is_valid_figure(fig):
                continue

            fig_info = {
                "figure_id": fig.get("figure_id", ""),
                "page_number": page_num or fig.get("page_number", ""),
                "figure_type": fig.get("figure_type", "figure"),
                "description": fig.get("description", ""),
                "confidence": fig.get("confidence", 0),
                "labels": fig.get("labels", []),
                "related_chapter": fig.get("related_chapter"),
                "related_topic": fig.get("related_topic"),
                "related_subtopic": fig.get("related_subtopic"),
            }
            indexed["all_figures"].append(fig_info)

            sub_id = fig.get("related_subtopic")
            if sub_id:
                indexed["by_subtopic"].setdefault(sub_id, []).append(fig_info)

            top_id = fig.get("related_topic")
            if top_id:
                indexed["by_topic"].setdefault(top_id, []).append(fig_info)

            chap_id = fig.get("related_chapter")
            if chap_id:
                indexed["by_chapter"].setdefault(chap_id, []).append(fig_info)

    return indexed


def generate_curriculum_html(data: list | dict, visual_data: dict | None = None) -> str:
    """
    Converts curriculum JSON data (and optional visual analysis JSON) into HTML.
    """
    if isinstance(data, dict):
        curriculums = [data]
    elif isinstance(data, list):
        curriculums = data
    else:
        raise ValueError("Invalid curriculum data format. Expected list or dict.")

    fig_index = index_figures(visual_data)

    html_parts = []
    
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curriculum & Visual Analysis View</title>
    <style>
        :root {
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --accent-color: #2563eb;
            --accent-light: #eff6ff;
            --border-color: #e2e8f0;
            --badge-bg: #f1f5f9;
            --id-bg: #f8fafc;
            --id-text: #475569;
            --theme-badge-bg: #e0e7ff;
            --theme-badge-text: #3730a3;
            --fig-badge-bg: #fef3c7;
            --fig-badge-text: #92400e;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
        }

        .header-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 2.5rem 2rem;
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            margin-bottom: 2rem;
        }

        .header-title {
            font-size: 2.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
            letter-spacing: -0.025em;
        }

        .metadata-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 1.25rem;
            margin-top: 1rem;
        }

        .metadata-item {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(4px);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
        }

        .metadata-label {
            color: #94a3b8;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: block;
            margin-bottom: 0.2rem;
        }

        .metadata-value {
            font-weight: 600;
            color: #f8fafc;
        }

        .chapter-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
        }

        .chapter-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .chapter-title-group {
            flex: 1;
        }

        .chapter-number {
            display: inline-block;
            color: var(--accent-color);
            font-weight: 700;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .chapter-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .tags-group {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .theme-badge {
            background-color: var(--theme-badge-bg);
            color: var(--theme-badge-text);
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .id-badge {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background-color: var(--id-bg);
            color: var(--id-text);
            border: 1px solid var(--border-color);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }

        .fig-count-badge {
            background-color: var(--fig-badge-bg);
            color: var(--fig-badge-text);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .topics-list {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .topic-item {
            background-color: #fafafa;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
        }

        .topic-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .topic-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-primary);
            flex: 1;
        }

        .subtopics-list {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            margin-top: 0.75rem;
            padding-left: 0.5rem;
        }

        .subtopic-item {
            background: var(--card-bg);
            border-left: 3px solid var(--accent-color);
            border-top: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
            border-radius: 0 6px 6px 0;
            padding: 0.85rem 1.1rem;
        }

        .subtopic-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.75rem;
            margin-bottom: 0.35rem;
            flex-wrap: wrap;
        }

        .subtopic-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .subtopic-description {
            font-size: 0.925rem;
            color: var(--text-secondary);
            margin-top: 0.4rem;
            line-height: 1.5;
        }

        .figure-box {
            margin-top: 0.75rem;
            background: #fffbe6;
            border: 1px solid #ffe58f;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
        }

        .figure-title {
            font-weight: 600;
            color: #d46b08;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .figure-desc {
            color: #595959;
        }
    </style>
</head>
<body>
    <div class="container">""")

    total_figures = len(fig_index["all_figures"])
    seen_img_hashes = set()

    for curr in curriculums:
        c_name = html.escape(str(curr.get("curriculum", "Curriculum")))
        grade = html.escape(str(curr.get("grade", "")))
        subject = html.escape(str(curr.get("subject", "")))
        chapters = curr.get("chapters", [])

        html_parts.append(f"""
        <header class="header-card">
            <h1 class="header-title">{subject or 'Curriculum'} Structure</h1>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Curriculum</span>
                    <span class="metadata-value">{c_name}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Subject</span>
                    <span class="metadata-value">{subject}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Grade</span>
                    <span class="metadata-value">{grade}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Chapters</span>
                    <span class="metadata-value">{len(chapters)}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Visual Figures Detected</span>
                    <span class="metadata-value">{total_figures}</span>
                </div>
            </div>
        </header>
        <main>""")

        for chap in chapters:
            chap_num = html.escape(str(chap.get("chapterNumber", "")))
            chap_title = html.escape(str(chap.get("chapterTitle", "")))
            chap_theme = html.escape(str(chap.get("theme", "")))
            chap_id = html.escape(str(chap.get("id", "")))
            topics = chap.get("topics", [])

            html_parts.append(f"""
            <section class="chapter-card">
                <div class="chapter-header">
                    <div class="chapter-title-group">
                        <span class="chapter-number">Chapter {chap_num}</span>
                        <h2 class="chapter-title">{chap_title}</h2>
                    </div>
                    <div class="tags-group">
                        {f'<span class="theme-badge">{chap_theme}</span>' if chap_theme else ''}
                        {f'<span class="id-badge">{chap_id}</span>' if chap_id else ''}
                    </div>
                </div>
                <div class="topics-list">""")

            for topic in topics:
                top_title = html.escape(str(topic.get("title", "")))
                top_id = html.escape(str(topic.get("topic_id", "")))
                subtopics = topic.get("subtopics", [])

                html_parts.append(f"""
                    <div class="topic-item">
                        <div class="topic-header">
                            <h3 class="topic-title">{top_title}</h3>
                            {f'<span class="id-badge">{top_id}</span>' if top_id else ''}
                        </div>
                        <div class="subtopics-list">""")

                for sub in subtopics:
                    sub_title = html.escape(str(sub.get("title", "")))
                    sub_desc = html.escape(str(sub.get("description", ""))) if sub.get("description") else ""
                    sub_id = html.escape(str(sub.get("sub_topic_id", "")))

                    # Find matching figures for this subtopic
                    sub_figures = fig_index["by_subtopic"].get(sub_id, [])

                    html_parts.append(f"""
                            <div class="subtopic-item">
                                <div class="subtopic-header">
                                    <h4 class="subtopic-title">{sub_title}</h4>
                                    {f'<span class="id-badge">{sub_id}</span>' if sub_id else ''}
                                </div>
                                {f'<p class="subtopic-description">{sub_desc}</p>' if sub_desc else ''}""")

                    for fig in sub_figures:
                        fig_id = html.escape(str(fig.get("figure_id", "")))
                        img_path = f"extracted_images/{fig_id}.png"
                        if not os.path.exists(img_path):
                            continue

                        # Deduplicate images by content hash so identical visuals are displayed only once
                        try:
                            with open(img_path, "rb") as f_img:
                                img_hash = hashlib.md5(f_img.read()).hexdigest()
                                if img_hash in seen_img_hashes:
                                    continue
                                seen_img_hashes.add(img_hash)
                        except Exception:
                            pass

                        fig_type = html.escape(str(fig.get("figure_type", "figure")))
                        fig_page = fig.get("page_number", "")
                        fig_desc = html.escape(str(fig.get("description", "")))
                        html_parts.append(f"""
                                <div class="figure-box">
                                    <div class="figure-title">
                                        <span>📷 {fig_type.upper()} ({fig_id})</span>
                                        <span class="id-badge">Page {fig_page}</span>
                                    </div>
                                    <div class="figure-img-container" style="margin: 0.75rem 0; text-align: center; background-color: #ffffff; border-radius: 6px; padding: 0.5rem; border: 1px solid #e2e8f0;">
                                        <img src="{img_path}" alt="{fig_desc or fig_id}" style="max-width: 100%; height: auto; max-height: 550px; border-radius: 4px; display: inline-block; object-fit: contain;">
                                    </div>
                                    <p class="figure-desc">{fig_desc}</p>
                                </div>""")

                    html_parts.append("""
                            </div>""")

                html_parts.append("""
                        </div>
                    </div>""")

            html_parts.append("""
                </div>
            </section>""")

        html_parts.append("""
        </main>""")

    html_parts.append("""
    </div>
</body>
</html>""")

    return "\n".join(html_parts)


def convert_json_to_html(input_json_path: str, output_html_path: str = "curriculum.html") -> str:
    """Reads JSON from file and writes HTML to output path."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Attempt to load visual analysis JSON if available
    visual_data = load_visual_data(input_json_path)

    # Extract images from PDF if visual data is available
    if visual_data:
        try:
            from image_extractor import extract_figures_from_pdf
            extract_figures_from_pdf(visual_data)
            
            # Persist updated visual_data back to disk
            path_obj = Path(input_json_path)
            visual_path = path_obj.parent / path_obj.name.replace("output_", "visual_analysis_")
            if visual_path.exists():
                with open(visual_path, "w", encoding="utf-8") as f_vis:
                    json.dump(visual_data, f_vis, indent=2)
        except Exception as e:
            print(f"Warning: Image extraction failed: {e}")

    html_content = generate_curriculum_html(data, visual_data)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path


def main():
    parser = argparse.ArgumentParser(description="Convert Curriculum JSON output to structured HTML file.")
    parser.add_argument(
        "json_file",
        nargs="?",
        default=None,
        help="Path to input curriculum JSON file (e.g. output_social-science.json)",
    )
    parser.add_argument(
        "html_file",
        nargs="?",
        default="curriculum.html",
        help="Path to output HTML file (default: curriculum.html)",
    )

    args = parser.parse_args()

    json_path = args.json_file
    if not json_path:
        candidates = list(Path(".").glob("output_*.json"))
        if candidates:
            json_path = str(candidates[0])
        elif os.path.exists("output_social-science.json"):
            json_path = "output_social-science.json"
        else:
            print("Error: No input JSON file provided and no default output_*.json found.")
            sys.exit(1)

    print(f"Reading JSON from {json_path}...")
    out_path = convert_json_to_html(json_path, args.html_file)
    print(f"Successfully generated HTML output: {out_path}")


if __name__ == "__main__":
    main()
