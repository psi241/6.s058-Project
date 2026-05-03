import os
import xml.etree.ElementTree as ET
import xmltodict
from PIL import Image

print("Currect path:", os.getcwd())

def annotation_loader(path, manga_name):
    path_to_ann_file = os.path.join(path, "Manga109/annotations/", f"{manga_name}.xml")
    with open(path_to_ann_file, 'r', encoding='utf-8') as file:
        file_xml = file.read()
        return xmltodict.parse(file_xml)

def get_book_list(path):
  '''
  Return a list of manga names
  '''
  
  with open(os.path.join(path, "Manga109/books.txt"), 'r', encoding='utf-8') as f:
    return [line.strip() for line in f.readlines()]



def get_character_list(dict_data):
  '''
  Return a list of ids and a list of names
  '''
  character_dict = dict_data['book']['characters']['character']
  return [ch['@id'] for ch in character_dict], [ch['@name'] for ch in character_dict]

def retrieve_page(path, manga_name, page: int, position = None, target_size = None):
  '''
  Arguments:
  manga_name: The name of the manga (e.g., 'Akuhamu').
  page: The page number as an integer.
  position: A tuple (xmin, ymin, xmax, ymax) for cropping.
  target_size: Optional target size for resizing the cropped image.
  '''
  page = int(page)
  img = Image.open(f'{path}/Manga109/images/{manga_name}/{page:03d}.jpg')
  # Crop image
  if position:
    cropped_img = img.crop(position)
    if target_size:
      cropped_img = cropped_img.resize(target_size)
    print(cropped_img)
  else:
    cropped_img = img
  return cropped_img

def get_crops_info_char(dict_data, char_id):
  '''
  Return crops for a given character (1) face (2) body, and positions
  '''
  pages_dict = dict_data['book']['pages']['page']
  char_faces = []
  char_bodies = []
  for page in pages_dict:
    faces = page.get('face')
    bodies = page.get('body')
    if isinstance(faces, dict):
      faces = [faces] 
    if isinstance(bodies, dict):
      faces = [faces] 
    if faces:
      print(faces)
      char_faces_page = [face for face in faces if face['@character'] == char_id]
      char_faces_page = [face | {'@index': page['@index']} for face in char_faces_page]
      for f in char_faces_page:
        f.pop('@character')
        x_min, y_min, x_max, y_max = int(f.pop('@xmin')), int(f.pop('@ymin')), int(f.pop('@xmax')), int(f.pop('@ymax'))
        f['position'] = (x_min, y_min, x_max, y_max)

      char_faces.extend(char_faces_page)
    if bodies:
      char_bodies_page = [body for body in bodies if body['@character'] == char_id]
      char_bodies_page = [body | {'@index': page['@index']} for body in char_bodies_page]
      for b in char_bodies_page:
        b.pop('@character')
        x_min, y_min, x_max, y_max = int(b.pop('@xmin')), int(b.pop('@ymin')), int(b.pop('@xmax')), int(b.pop('@ymax'))
        b['position'] = (x_min, y_min, x_max, y_max)
      char_bodies.extend(char_bodies_page)

  return char_faces, char_bodies

def get_location(dict_data, obj):
  '''
  Get all faces locations for a given page
  Return a dictionary with page as key and a list of faces as value. Each face is a dictionary with keys: character, position, and page index.

  obj can be 'face' or 'body'
  Use this to compute IOU
  '''
  pages_dict = dict_data['book']['pages']['page']
  out = {}
  for page in pages_dict:
    out[page['@index']] = []
    if obj == 'face':
      faces = page.get('face')
      if isinstance(faces, dict):
        faces = [faces]
      if faces:
        for face in faces:
          out[page['@index']].append({
              'character': face['@character'],
              'position': (int(face['@xmin']), int(face['@ymin']), int(face['@xmax']), int(face['@ymax']))
          })
    elif obj == 'body':   
      bodies = page.get('body')
      if isinstance(bodies, dict):
        bodies = [bodies]
      if bodies:
        for body in bodies:
          out[page['@index']].append({
              'character': body['@character'],
              'position': (int(body['@xmin']), int(body['@ymin']), int(body['@xmax']), int(body['@ymax']))
          })
  return out

def create_data(data_dicts):
  '''
  Make a tensor of samples
  '''
  pass

if __name__ == "__main__":
    dotenv.load_dotenv()
    DATA_PATH = os.getenv("DATASET_PATH")

    data = annotation_loader(DATA_PATH, "AisazuNihaIrarenai")
    print(data['book']['pages'])