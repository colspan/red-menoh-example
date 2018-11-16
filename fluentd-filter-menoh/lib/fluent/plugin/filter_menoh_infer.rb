require "fluent/plugin/filter"
require "fluent/config/error"

require "json"
require "menoh"
require "rmagick"

# onnx variable name
MNIST_IN_NAME = "139900320569040".freeze
MNIST_OUT_NAME = "139898462888656".freeze

# onnx variable name
CONV1_1_IN_NAME = "Input_0".freeze
FC6_OUT_NAME = "Gemm_0".freeze
SOFTMAX_OUT_NAME = "Softmax_0".freeze
TOP_K = 5

module Fluent
  class MenohFilter < Fluent::Plugin::Filter
    Fluent::Plugin.register_filter("menoh_infer", self)

    def configure(conf)
      super
      setup_vgg(conf)
    end

    def setup_mnist(conf)
      @input_shape = {
        channel_num: 1,
        width: 28,
        height: 28,
      }

      # load ONNX file
      @onnx_obj = Menoh::Menoh.new "./data/mnist.onnx"

      # model options for model
      @model_opt = {
        backend: "mkldnn",
        input_layers: [
          {
            name: MNIST_IN_NAME,
            dims: [
              1,
              @input_shape[:channel_num],
              @input_shape[:height],
              @input_shape[:width],
            ],
          },
        ],
        output_layers: [MNIST_OUT_NAME],
      }
      @categories = (0..9).to_a
    end

    def setup_vgg(conf)
      @input_shape = {
        channel_num: 3,
        width: 224,
        height: 224,
      }

      # load ONNX file
      @onnx_obj = Menoh::Menoh.new "./data/VGG16.onnx"

      # model options for model
      @model_opt = {
        backend: "mkldnn",
        input_layers: [
          {
            name: CONV1_1_IN_NAME,
            dims: [
              1,
              @input_shape[:channel_num],
              @input_shape[:height],
              @input_shape[:width],
            ],
          },
        ],
        output_layers: [FC6_OUT_NAME, SOFTMAX_OUT_NAME],
      }
      # load category definition
      @categories = File.read("./data/synset_words.txt").split("\n")

      @rgb_offset = {
        R: 123.68,
        G: 116.779,
        B: 103.939
      }
      
    end

    def start
      super
      @model = @onnx_obj.make_model @model_opt
    end

    def preprocess_mnist(es)
      batch = []
      meta = []
      es.each do |time, record|
        image = image.resize_to_fill(@input_shape[:width], @input_shape[:height])
        batch << image.export_pixels(0, 0, image.columns, image.rows, "i").map { |pix| pix / 256 } # for MNIST
        meta << {
          time: time
        }
      end
      image_set = [
        {
          name: MNIST_IN_NAME,
          data: batch.flatten
        }
      ]
      [image_set, meta]
    end

    def preprocess_vgg(es)
      batch = []
      meta = []
      es.each do |time, record|
        img_base64 = JSON.load(record["message"])["img"].gsub("data:image/jpeg;base64,", "")
        image = Magick::Image.read_inline(img_base64).first
        image = image.resize_to_fill(@input_shape[:width], @input_shape[:height])
        batch << 'RGB'.split('').map do |color|
          image.export_pixels(0, 0, image.columns, image.rows, color).map do |pix|
            pix / 256 - @rgb_offset[color.to_sym]
          end
        end.flatten
        meta << {
          time: time
        }
      end
      image_set = [
        {
          name: CONV1_1_IN_NAME,
          data: batch.flatten
        }
      ]
      [image_set, meta]
    end


    def postprocess_mnist(meta, detected_results)
      new_es = MultiEventStream.new
      top_k = 1
      output = []
      layer_result = detected_results.find { |x| x[:name] == MNIST_OUT_NAME }
      layer_result[:data].zip(es).each do |image_result, image_filepath|
        # sort by score
        sorted_result = image_result.zip(@categories).sort_by { |x| -x[0] }

        # display result
        sorted_result[0, top_k].each do |score, category|
          puts "#{category} : #{score}"
        end
      end
      output
    end

    def postprocess_vgg(meta, detected_results)
      new_es = MultiEventStream.new
      top_k = 5
      layer_result = detected_results.find { |x| x[:name] == SOFTMAX_OUT_NAME }
      layer_result[:data].zip(meta).each do |image_result, m|
        # sort by score
        sorted_result = image_result.zip(@categories).sort_by { |x| -x[0] }
        # display result
        result = []
        sorted_result[0...top_k].each do |score, category|
          result << {time: m[:time], category: category, score: score}
        end
        new_es.add(m[:time], result)
      end
      new_es
    end

    def filter_stream(tag, es)
      new_es = nil
      begin
        # prepare dataset
        image_set, meta = preprocess_vgg(es)

        # execute inference
        detected_results = @model.run image_set

        # postprocess_mnist(es, detected_results)
        new_es = postprocess_vgg(meta, detected_results)
      rescue StandardError => e
        router.emit_error_event(tag, nil, nil, e)
      end
      new_es
    end
  end
end
