module Paperclip
  class Cropper < Thumbnail

    def transformation_command
      if crop_command
        crop_command + super.join(' ').sub(/ -crop \S+/, '').split(' ')
      else
        super
      end
    end


    def crop_command
      target = @attachment.instance

      if target.cropping?(@attachment.name)
        w = target.send :"#{@attachment.name}_crop_w", sanitize: false
        h = target.send :"#{@attachment.name}_crop_h", sanitize: false
        x = target.send :"#{@attachment.name}_crop_x", sanitize: false
        y = target.send :"#{@attachment.name}_crop_y", sanitize: false
        ["-crop", "#{w}x#{h}+#{x}+#{y}"]
      end
    end

  end
end
