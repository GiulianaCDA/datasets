require "paperclip"

module Paperclip
  class Cropper < Thumbnail
    def transformation_command
      crop_command.nil? ? super : (crop_command + super.join(' ').sub(/ -crop \S+/, '').split(' '))
    end

    private

    def crop_command
      target = @attachment.instance
      if target.respond_to?(:cropping?) && target.cropping?(@attachment.name)
        w = target.send :"#{@attachment.name}_crop_w"
        h = target.send :"#{@attachment.name}_crop_h"
        x = target.send :"#{@attachment.name}_crop_x"
        y = target.send :"#{@attachment.name}_crop_y"
        ["-crop", "#{w}x#{h}+#{x}+#{y}"]
      end
    end
  end
end
