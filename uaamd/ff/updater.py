import os
import tarfile
import requests
from bs4 import BeautifulSoup
import click

MACKERELL_LAB_URL = "https://mackerell.umaryland.edu/charmm_ff.shtml"
DOWNLOAD_DIR = os.path.expanduser("~/.uaamd/ff")

def get_latest_charmm36_url():
    """Scrapes the MacKerell lab website to find the latest CHARMM36 GROMACS port."""
    try:
        response = requests.get(MACKERELL_LAB_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for links containing "charmm36" and ".tgz" or ".tar.gz" and ".ff."
        # This identifies the GROMACS ports
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and '.ff.tgz' in href and 'charmm36' in href.lower():
                # Construct absolute URL
                if href.startswith('http'):
                    return href
                elif href.startswith('/'):
                    return f"https://mackerell.umaryland.edu{href}"
                else:
                    return f"https://mackerell.umaryland.edu/{href}"

        # Fallback if specific GROMACS port is not found, but this is specific to standard naming
        # Currently, typically named something like charmm36-jul2022.ff.tgz
    except Exception as e:
        click.echo(f"Error fetching from {MACKERELL_LAB_URL}: {e}")
        return None

    return None

def download_and_extract(url, target_dir):
    """Downloads a file from a URL and extracts it into the target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(target_dir, filename)

    click.echo(f"Downloading {url} to {filepath}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        click.echo("Download complete. Extracting...")

        if filepath.endswith('.tgz') or filepath.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=target_dir)

        click.echo(f"Successfully extracted to {target_dir}")
        return True
    except Exception as e:
        click.echo(f"Error during download/extraction: {e}")
        return False

def update_charmm36():
    """Updates the CHARMM36 force field."""
    click.echo("Checking for latest CHARMM36 force field (GROMACS format)...")
    url = get_latest_charmm36_url()

    if not url:
        click.echo("Failed to find a suitable CHARMM36 download link.")
        # Fallback to a known recent version directly if scraping fails
        url = "https://mackerell.umaryland.edu/download/charmm36/charmm36-jul2022.ff.tgz"
        click.echo(f"Falling back to known URL: {url}")

    success = download_and_extract(url, DOWNLOAD_DIR)

    if success:
        click.echo("Force field update completed successfully.")
    else:
        click.echo("Force field update failed.")
