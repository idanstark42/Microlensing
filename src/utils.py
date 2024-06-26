class Value:
  def __init__(self, value, error):
    self.value = value
    self.error = error

  def __str__(self):
    return f"{self.value} ± {self.error}"
  
  def __repr__(self):
    return f"Value({self.value}, {self.error})"