require 'json'
require 'narray'
require 'menoh'
require 'open-uri'
require 'rmagick'

# https://qiita.com/shima_x/items/79f4fd33d24ea338d8b2
def resize_and_pad(image, width, height)
  # change_geometryを通さないとpaddingが効いてくれない
  image.change_geometry("#{width}x#{height}") do |cols,rows,img|
    # !を付けると破壊的にimgを上書きする
    img.resize_to_fit!(cols,rows)
    # 背景色を白にする
    img.background_color = '#808080'
    # ImageMagickの仕様で位置合わせの座標にマイナスが付与される
    # そのため第3, 第4引数にはマイナスを付ける必要がある
    img.extent(width, height, -(width-cols)/2, -(height-rows)/2)
  end
end

def decode(out, anchors, n_fg_class, thresh)
  out_h = out[0].length
  out_w = out[0][0].length
  score = nil
  # reshape
  out = out.each_slice(4 + 1 + n_fg_class).to_a
  bbox = []
  (0...out_h).each do |y|
    (0...out_w).each do |x|
      (0...anchors.length).each do |a|
        # WIP still broken value
        obj = out[a][4][y][x]
        conf = out[a][(4 + 1)...(4 + 1 + n_fg_class)]
        anc_y = y.to_f + sigmoid(out[a][0][y][x])
        anc_x = x.to_f + sigmoid(out[a][1][y][x])
        anc_h = anchors[a][0] * Math.exp(out[a][2][y][x])
        anc_w = anchors[a][1] * Math.exp(out[a][3][y][x])

        obj = sigmoid(obj)
        score = conf.map { |d| Math.exp(d[y][x]) }
        sum = score.inject(:+)
        # p sum
        score.map! { |d| d * obj / sum }
        # p score
        (0...n_fg_class).each do |lb|
          next unless score[lb] >= thresh

          bbox.push(
            top: (anc_y - anc_h / 2.0) / out_h.to_f,
            left: (anc_x - anc_w / 2.0) / out_w.to_f,
            bottom: (anc_y + anc_h / 2.0) / out_h.to_f,
            right: (anc_x + anc_w / 2.0) / out_w.to_f,
            label: lb,
            score: score[lb]
          )
        end
      end
    end
  end
  bbox
end

def sigmoid(x)
  1. / (1. + Math.exp(-x))
end

def suppress(bboxes, thresh)
  bboxes.sort { |a, b| b[:score] <=> a[:score] }
        .uniq { |a| a[:label] }
        .delete_if { |a| a[:score] < thresh }
end

# download dependencies
def download_file(url, output)
  return if File.exist? output

  puts "downloading... #{url}"
  File.open(output, 'wb') do |f_output|
    open(url, 'rb') do |f_input|
      f_output.write f_input.read
    end
  end
end
download_file('https://github.com/Hakuyume/menoh-yolo/releases/download/assets/yolo_v2_voc0712.onnx', './data/yolo_v2_voc0712.onnx')
download_file('https://github.com/Hakuyume/menoh-yolo/releases/download/assets/yolo_v2_voc0712.json', './data/yolo_v2_voc0712.json')
download_file('https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg', './data/Light_sussex_hen.jpg')
download_file('https://upload.wikimedia.org/wikipedia/commons/f/fd/FoS20162016_0625_151036AA_%2827826100631%29.jpg', './data/honda_nsx.jpg')
download_file('https://github.com/pjreddie/darknet/raw/master/data/dog.jpg', './data/dog.jpg')

# load config
config = JSON.load(File.open('./data/yolo_v2_voc0712.json', 'r').read)

# load dataset
image_list = [
  # './data/Light_sussex_hen.jpg',
  # './data/honda_nsx.jpg',
  './data/dog.jpg'
]
input_shape = {
  channel_num: 3,
  width: config['insize'],
  height: config['insize']
}

# load ONNX file
onnx_obj = Menoh::Menoh.new './data/yolo_v2_voc0712.onnx'

# model options for model
model_opt = {
  backend: 'mkldnn',
  input_layers: [
    {
      name: config['input'],
      dims: [
        image_list.length,
        input_shape[:channel_num],
        input_shape[:height],
        input_shape[:width]
      ]
    }
  ],
  output_layers: [config['output']]
}
# make model for inference under 'model_opt'
model = onnx_obj.make_model model_opt

# prepare dataset
image_set = [
  {
    name: config['input'],
    data: image_list.map do |image_filepath|
      image = Magick::Image.read(image_filepath).first
      image = resize_and_pad(image, input_shape[:width], input_shape[:height])
      'RGB'.split('').map do |color|
        image.export_pixels(0, 0, image.columns, image.rows, color).map { |pix| pix.to_f / 65_536.0 }
      end.flatten
    end.flatten
  }
]

# execute inference
inferenced_results = model.run image_set

layer_result = (inferenced_results.find { |x| x[:name] == config['output'] })
layer_result[:data].zip(image_list, image_set.first[:data]).each do |image_result, image_filepath|
  puts "=== Result for #{image_filepath} ==="
  image_buffer = Magick::Image.read(image_filepath).first
  image_buffer = resize_and_pad(image_buffer, input_shape[:width], input_shape[:height])
  org_w = image_buffer.columns
  org_h = image_buffer.rows
  p [org_w, org_h]
  scale = 128#/org_w.to_f# / 13.0
  p scale
  bboxes = decode(image_result, config['anchors'], config['label_names'].length, 0.5)
  bboxes = suppress(bboxes, 0.45)
  bboxes.each do |bbox|
    p [bbox[:score], config['label_names'][bbox[:label]]]
    p [bbox[:top], bbox[:left], bbox[:bottom], bbox[:right]]
    draw = Magick::Draw.new
    draw.fill('#ffffff')
    draw.rectangle(
      bbox[:top] * scale + org_h / 2.0,
      bbox[:left] * scale + org_w / 2.0,
      bbox[:bottom] * scale + org_h / 2.0,
      bbox[:right] * scale + org_w / 2.0
    )
    draw.draw(image_buffer)
  end
  image_buffer.write(image_filepath + 'out.png')
end
