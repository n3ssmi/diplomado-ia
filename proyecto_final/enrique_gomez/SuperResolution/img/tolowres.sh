find ./hi/ -name "*.jpg" -exec magick {} -verbose -quality 5 {}_low.jpg \;
