import requests
from bs4 import BeautifulSoup

from src.utils import Value, I

class Event:
  def __init__(self, year, id):
    self.year = year
    self.id = id

    self.download_metadata()
    self.download_images()
    self.download_data()

    self.normalize_time()

  def download_data(self):
    data_str = requests.get(f"https://www.astrouw.edu.pl/ogle/ogle4/ews/{self.year}/{self.id}/phot.dat").text
    self.data = [{'t': float(raw[0]), 'I': I(Value(float(raw[1]), float(raw[2])), self.metadata['I0']) } for raw in [line.split() for line in data_str.split('\n') if len(line) > 0]]
  
  def download_images(self):
    self.large_image = requests.get(f"https://www.astrouw.edu.pl/ogle/ogle4/ews/{self.year}/{self.id}/lcurve.gif").content
    self.small_image = requests.get(f"https://www.astrouw.edu.pl/ogle/ogle4/ews/{self.year}/{self.id}/lcurve_s.gif").content

  def download_metadata(self):
    # scrape the metadata from the event page
    event_page_url = f"https://ogle.astrouw.edu.pl/ogle4/ews/{self.year}/{self.id}.html"
    event_page = requests.get(event_page_url).text
    
    # the metadata is in the third table of the page, use BeautifulSoup to parse it
    soup = BeautifulSoup(event_page, 'html.parser')
    metadata_table = soup.find_all('table')[2]
    metadata_lines = [line.split() for line in metadata_table.text.split('\n') if line != '' and line[0] != '(']
    self.metadata = { line[0]: Value(float(line[1]), float(line[3])) for line in metadata_lines }

  def normalize_time(self):
    self.time_shift = min([datum['t'] for datum in self.data])
    for datum in self.data:
      datum['t'] -= self.time_shift

    self.metadata['Tmax'].value -= self.time_shift

  def __str__(self):
    return f"{self.year} {self.id}" + '\n\n' + '\n'.join([key + ":\t"  + str(value) for key, value in self.metadata.items()])

  def points_around_peak (self, time_window):
    return [datum for datum in self.data if abs(datum['t'] - self.metadata['Tmax'].value) < time_window / 2]