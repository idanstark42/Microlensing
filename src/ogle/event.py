import requests
from bs4 import BeautifulSoup

from src.utils import Value

class Event:
  def __init__(self, year, id):
    self.year = year
    self.id = id

  def data_url(self):
    return f"https://www.astrouw.edu.pl/ogle/ogle4/ews/{self.year}/{self.id}/phot.dat"
  
  def image_url(self):
    return f"https://www.astrouw.edu.pl/ogle/ogle4/ews/{self.year}/{self.id}/lcurve_s.gif"
  
  def download_data(self):
    data_str = requests.get(self.data_url()).text
    self.data = [{ 't': float(raw[0]), 'm': Value(float(raw[1]), (raw[2])) } for raw in [line.split() for line in data_str.split('\n') if len(line) > 0]]
  
  def download_image(self):
    self.image = requests.get(self.image_url()).content

  def download_metadata(self):
    # scrape the metadata from the event page
    event_page_url = f"https://ogle.astrouw.edu.pl/ogle4/ews/{self.year}/{self.id}.html"
    event_page = requests.get(event_page_url).text
    
    # the metadata is in the third table of the page, use BeautifulSoup to parse it
    soup = BeautifulSoup(event_page, 'html.parser')
    metadata_table = soup.find_all('table')[2]
    metadata_lines = [line.split() for line in metadata_table.text.split('\n') if line != '' and line[0] != '(']
    self.metadata = { line[0]: Value(float(line[1]), float(line[3])) for line in metadata_lines }

  def __str__(self):
    if (not self.metadata):
      self.download_metadata()

    return f"{self.year} {self.id}" + '\n\n' + '\n'.join([key + ":\t"  + str(value) for key, value in self.metadata.items()])