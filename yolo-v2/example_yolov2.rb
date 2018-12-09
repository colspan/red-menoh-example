require 'json'
require 'narray'
require 'open-uri'
require 'rmagick'

require_relative 'menoh-yolov2'

dataset = [
  {
    url: 'https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg',
    path: './data/Light_sussex_hen.jpg'
  },
  {
    url: 'https://upload.wikimedia.org/wikipedia/commons/f/fd/FoS20162016_0625_151036AA_%2827826100631%29.jpg',
    path: './data/honda_nsx.jpg'
  },
  {
    url: 'https://github.com/pjreddie/darknet/raw/master/data/dog.jpg',
    path: './data/dog.jpg'
  }
]

# download dependencies
dataset.each { |x| Menoh::Util.download_file(x[:url], x[:path]) }

# load dataset
img_list = dataset.map { |x| x[:path] }
img_bufs = img_list.map do |path|
  Magick::Image.read(path).first
end

yolov2 = Menoh::YOLOv2.new(batch_size: img_list.length)

results = yolov2.predict(img_bufs)

results.zip(img_list, img_bufs).each do |result, path, img_buf|
  puts "=== Result for #{path} ==="

  result[:objects].each do |object|
    puts " label : #{object[:label]}, score : #{object[:score]}"
    draw = Magick::Draw.new
    draw.stroke('#ffff00')
    draw.fill('Transparent')
    draw.rectangle(*object[:bbox])
    draw.draw(img_buf)
  end
  img_buf.write("data/#{File.basename(path, File.extname(path))}_detected.png")
end
