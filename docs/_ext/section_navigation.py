"""Use the root toctree as section tabs and show one section in Furo's sidebar."""

from bs4 import BeautifulSoup


def section_navigation(app, pagename, templatename, context, doctree):
    env = app.env
    roots = env.toctree_includes.get(env.config.root_doc, [])

    def contains(root, page):
        return root == page or any(
            contains(child, page) for child in env.toctree_includes.get(root, [])
        )

    active = next((root for root in roots if contains(root, pagename)), None)
    if pagename == env.config.root_doc:
        active = roots[0] if roots else None
    context["docs_sections"] = [
        {"doc": root, "title": env.titles[root].astext(), "active": root == active}
        for root in roots
    ]
    context["docs_section"] = env.titles[active].astext() if active else "Documentation"
    context["docs_section_root"] = active

    # Furo has added accessible disclosure controls and current-page markers.
    # Keep those, selecting only the active section's children.
    soup = BeautifulSoup(context.get("furo_navigation_tree", ""), "html.parser")
    link = soup.find("a", href=context["pathto"](active)) if active else None
    tree = link.parent.find("ul", recursive=False) if link else None
    if tree:
        # Deep subtrees (the API module listing) stay collapsed to their entry
        # unless the current page lives inside them.
        for item in tree.select(":scope > li.has-children"):
            if "current" in item.get("class", []):
                continue
            for child in list(item.children):
                if getattr(child, "name", None) in {"ul", "input", "label"}:
                    child.decompose()
            item["class"] = ["toctree-l2"]
        for item in tree.select("li.current-page > a"):
            item["aria-current"] = "page"
    if tree:
        context["docs_section_tree"] = str(tree)
    elif active:
        context["docs_section_tree"] = ""
    else:
        # Pages outside every section keep the full tree rather than none.
        context["docs_section_tree"] = context.get("furo_navigation_tree", "")


def setup(app):
    app.connect("html-page-context", section_navigation, priority=800)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
