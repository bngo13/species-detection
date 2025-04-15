for file in $(find . -name *.tif); do
  magick "${file}" -resize 50% "./${file%.tif}.png"
done
