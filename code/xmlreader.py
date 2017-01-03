import xml.etree.ElementTree as ET

tree = ET.parse('a_fairy_blunder.xml')
root = tree.getroot()

text = root.iter('text')
# for t in text:
	# print(t.text)

story_dict = {}

for child in root.iter('annotationSet'):
	frame = child.get('frameName')
	print(frame)

# 	if frame in story_dict.keys():
# 		story_dict[frame] = story_dict[frame] + 1
# 	else:
# 		story_dict[frame] = 1

# 	# print(child.get('frameName'))

# for key in sorted(story_dict.keys()):
# 	print(key, story_dict[key])