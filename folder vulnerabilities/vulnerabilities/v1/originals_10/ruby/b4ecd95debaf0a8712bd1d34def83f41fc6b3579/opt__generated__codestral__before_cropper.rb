require "paperclip"

module Paperclip
  class Cropper < Thumbnail
    attr_reader :attachment, :instance # Add attribute reader for attachment and instance

    def transformation_command
      if crop_command
        crop_command + super.join(' ').sub(/ -crop \S+/, '').split(' ')
      else
        super
      end
    end

    def crop_command
      target = @instance # Use attribute reader instead of send

      if target.cropping?(attachment.name)
        w = target.public_send("#{attachment.name}_crop_w")
        h = target.public_send("#{attachment.name}_crop_h")
        x = target.public_send("#{attachment.name}_crop_x")
        y = target.public_send("#{attachment.name}_crop_y")
        ["-crop", "#{w}x#{h}+#{x}+#{y}"]
      end
    end
  end
end
