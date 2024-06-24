from event import Event

def load_event (year, id):
  event = Event(year, id)
  event.download_data()
  event.download_image()
  return event