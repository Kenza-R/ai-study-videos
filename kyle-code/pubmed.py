"""
Fetch paper content from PubMed Central (PMC).
"""
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import pubmed_parser as pp


class PMCNotFoundError(Exception):
    """Raised when a paper is not available in PubMed Central."""
    pass


def fetch_paper(paper_id: str, output_dir: str) -> Dict:
    """
    Fetch a paper from PubMed Central and save to output directory.

    Args:
        paper_id: PubMed ID (e.g., "12345678") or PMC ID (e.g., "PMC8675309")
        output_dir: Directory to save paper data

    Returns:
        Dict containing paper metadata, full text, and figures

    Raises:
        PMCNotFoundError: If paper is not in PMC
    """
    # Step 1: Determine if input is PMID or PMCID
    if paper_id.upper().startswith("PMC"):
        # Already a PMCID
        pmcid = paper_id.upper()
        pmid = None  # We'll try to get it from the XML later
    else:
        # It's a PMID, look up the PMCID
        pmid = paper_id
        pmcid = get_pmcid(pmid)
        if not pmcid:
            raise PMCNotFoundError(
                f"PMID {pmid} is not available in PubMed Central. "
                "This tool only works with open-access papers in PMC."
            )

    # Step 2: Fetch full text XML from PMC
    xml_path = download_pmc_xml(pmcid, output_dir)

    # Step 3: Parse XML to extract content using pubmed_parser
    paper_data = parse_pmc_xml(xml_path, pmid, pmcid)

    # Step 4: Save to output directory
    output_path = Path(output_dir)
    json_path = output_path / "paper.json"
    with open(json_path, 'w') as f:
        json.dump(paper_data, f, indent=2)

    return paper_data


def get_pmcid(pmid: str) -> Optional[str]:
    """
    Look up PMCID from PMID using PubMed API.

    Returns:
        PMCID string (e.g., "PMC8675309") or None if not in PMC
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"

    with urllib.request.urlopen(url) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)

    # Look for PMC ID in ArticleIdList
    for article_id in root.findall(".//ArticleId"):
        if article_id.get("IdType") == "pmc":
            pmc_id = article_id.text
            # Ensure it has PMC prefix
            if not pmc_id.startswith("PMC"):
                pmc_id = f"PMC{pmc_id}"
            return pmc_id

    return None


def download_pmc_xml(pmcid: str, output_dir: str) -> str:
    """
    Download full text XML from PMC to output directory.

    Args:
        pmcid: PMC ID (e.g., "PMC8675309")
        output_dir: Directory to save XML file

    Returns:
        Path to downloaded XML file
    """
    # Strip "PMC" prefix if present for the API call
    pmc_number = pmcid.replace("PMC", "")
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_number}&retmode=xml"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_path = output_path / "paper.xml"

    with urllib.request.urlopen(url) as response:
        xml_data = response.read()

    with open(xml_path, 'wb') as f:
        f.write(xml_data)

    return str(xml_path)


def parse_pmc_xml(xml_path: str, pmid: Optional[str], pmcid: str) -> Dict:
    """
    Parse PMC XML to extract paper content using pubmed_parser.

    Returns:
        Dictionary with structure:
        {
            "pmid": "...",
            "pmcid": "PMC...",
            "title": "...",
            "full_text": "...",
            "figures": [
                {"id": "fig1", "url": "https://...", "caption": "..."}
            ]
        }
    """
    # Use pubmed_parser to parse the XML
    parsed = pp.parse_pubmed_xml(xml_path)

    # Extract title
    title = parsed.get('full_title', 'Unknown')

    # If pmid wasn't provided, try to extract it from parsed data
    if not pmid:
        pmid = parsed.get('pmid', None)

    # Extract full text from abstract and body
    abstract = parsed.get('abstract', '')

    # Parse paragraphs from XML for full text
    paragraphs = pp.parse_pubmed_paragraph(xml_path, all_paragraph=True)

    # Concatenate all paragraph text
    body_text = '\n\n'.join([p['text'] for p in paragraphs if p.get('text')])

    # Combine abstract and body
    full_text = f"{abstract}\n\n{body_text}" if abstract else body_text

    # Extract figures
    # Parse the XML to get figures (pubmed_parser doesn't have a direct figure parser)
    root = ET.parse(xml_path).getroot()
    figures = []

    # Search for fig elements (namespace-agnostic)
    for fig in root.iter():
        if fig.tag.endswith('}fig') or fig.tag == 'fig':
            fig_id = fig.get('id', '')

            # Get caption
            caption = ""
            for elem in fig.iter():
                if elem.tag.endswith('}caption') or elem.tag == 'caption':
                    caption = "".join(elem.itertext())
                    break

            # Get image URL (try graphic element)
            for elem in fig.iter():
                if elem.tag.endswith('}graphic') or elem.tag == 'graphic':
                    # Try different xlink attribute formats
                    xlink_href = (
                        elem.get("{http://www.w3.org/1999/xlink}href") or
                        elem.get("href") or
                        ""
                    )
                    if xlink_href:
                        # Construct full URL to PMC image
                        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/{xlink_href}"
                        figures.append({
                            "id": fig_id,
                            "url": url,
                            "caption": caption.strip()
                        })
                        break  # Only take first graphic per figure

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "title": title.strip(),
        "full_text": full_text.strip(),
        "figures": figures
    }
