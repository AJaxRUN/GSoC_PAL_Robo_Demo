def rename_elements(element, suffix):
    if element.tag == "body" and element.get("name"):
        element.set("name", f"{element.get('name')}_{suffix}")
    if element.tag == "joint" and element.get("name"):
        element.set("name", f"{element.get('name')}_{suffix}")
    if element.tag == "geom" and element.get("name"):
        element.set("name", f"{element.get('name')}_{suffix}")
    if element.tag == "site" and element.get("name"):
        element.set("name", f"{element.get('name')}_{suffix}")
    if element.tag == "camera":
        element.getparent().remove(element)
    for child in element:
        rename_elements(child, suffix)