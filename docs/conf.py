import matplotlib.font_manager as fm
fonts = fm.findSystemFonts()
print([[str(font), fm.FontProperties(fname=font).get_name()] for font in fonts])

import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
plt.xlabel("時間")
plt.savefig('test.jpg')