for file in $(find . -name *.tif); do
  magick "${file}" -resize 35% "../smoldata/${file%.tif}.png"
done
