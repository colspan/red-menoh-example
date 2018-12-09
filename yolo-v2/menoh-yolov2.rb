require 'menoh'

module Menoh
  module Util
    def self.download_file(url, output)
      return if File.exist? output

      puts "downloading... #{url}"
      File.open(output, 'wb') do |f_output|
        open(url, 'rb') do |f_input|
          f_output.write f_input.read
        end
      end
    end

    # base implemantation is from
    # https://qiita.com/shima_x/items/79f4fd33d24ea338d8b2
    def self.resize_and_pad(image, width, height)
      # change_geometryを通さないとpaddingが効いてくれない
      image.change_geometry("#{width}x#{height}") do |cols, rows, img|
        # !を付けると破壊的にimgを上書きする
        output = img.resize_to_fit(cols, rows)
        # 背景色を白にする
        output.background_color = '#808080'
        # ImageMagickの仕様で位置合わせの座標にマイナスが付与される
        # そのため第3, 第4引数にはマイナスを付ける必要がある
        output.extent(width, height, -(width - cols) / 2, -(height - rows) / 2)
      end
    end

    def self.sigmoid(x)
      1. / (1. + Math.exp(-x))
    end
  end
  class YOLOv2
    def initialize(
      batch_size: 10,
      onnx_path: './data/yolo_v2_voc0712.onnx',
      config_path: './data/yolo_v2_voc0712.json'
    )

      # load config
      @config = JSON.load(File.open(config_path, 'r').read)

      # set dimension
      @input_shape = {
        channel_num: 3,
        width: @config['insize'],
        height: @config['insize']
      }
      @batch_size = batch_size

      # load ONNX file
      @onnx_obj = Menoh.new(onnx_path)

      # model options for model
      @model_opt = {
        backend: 'mkldnn',
        input_layers: [
          {
            name: @config['input'],
            dims: [
              batch_size,
              @input_shape[:channel_num],
              @input_shape[:height],
              @input_shape[:width]
            ]
          }
        ],
        output_layers: [@config['output']]
      }
      # make model for inference under 'model_opt'
      @model = @onnx_obj.make_model(@model_opt)
    end

    def predict(images, threshold: 0.50, sup: 0.45)
      # TODO: raise if images.length > batch_size

      # open images as Magick::Image
      original_set = images.map do |image|
        image = Magick::Image.read(image).first if image.instance_of?(String)
        # TODO: raise if image is not Magick::Image
        # TODO convert if image is not 3 channel image
        image
      end

      # prepare dataset
      image_set = [{
        name: @config['input'],
        data: original_set.map do |image|
          image = Util.resize_and_pad(
            image,
            @input_shape[:width],
            @input_shape[:height]
          )
          'RGB'.split('').map do |color|
            image.export_pixels(
              0, 0, image.columns, image.rows, color
            ).map { |pix| pix.to_f / 65_536.0 }
          end.flatten
        end.flatten
      }]

      # execute inference
      raw_results = @model.run image_set

      results = []
      layer_result = (raw_results.find { |x| x[:name] == @config['output'] })
      layer_result[:data].zip(original_set).each do |image_result, img_org|
        objects = []
        org_w = img_org.columns.to_f
        org_h = img_org.rows.to_f

        # calculate scale
        scale = [
          org_w / @input_shape[:width],
          org_h / @input_shape[:height]
        ].max
        scale_w = @input_shape[:width] * scale
        scale_h = @input_shape[:height] * scale

        # decode
        bboxes = decode(
          image_result,
          @config['anchors'],
          @config['label_names'].length,
          threshold
        )
        bboxes = suppress(bboxes, sup)
        bboxes.each do |bbox|
          objects.push(
            label: @config['label_names'][bbox[:label]],
            score: bbox[:score],
            bbox: [
              (bbox[:left] - 0.5) * scale_w + org_w / 2.0,
              (bbox[:top] - 0.5) * scale_h + org_h / 2.0,
              (bbox[:right] - 0.5) * scale_w + org_w / 2.0,
              (bbox[:bottom] - 0.5) * scale_h + org_h / 2.0
            ]
          )
        end
        results.push(
          objects: objects
        )
      end
      results
    end

    private

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
            obj = out[a][4][y][x]
            conf = out[a][(4 + 1)...(4 + 1 + n_fg_class)]
            anc_y = y.to_f + Util.sigmoid(out[a][0][y][x])
            anc_x = x.to_f + Util.sigmoid(out[a][1][y][x])
            anc_h = anchors[a][0] * Math.exp(out[a][2][y][x])
            anc_w = anchors[a][1] * Math.exp(out[a][3][y][x])

            obj = Util.sigmoid(obj)
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

    def suppress(bboxes, thresh)
      bboxes.sort { |a, b| b[:score] <=> a[:score] }
            .uniq { |a| a[:label] }
            .delete_if { |a| a[:score] < thresh }
    end
  end
end
