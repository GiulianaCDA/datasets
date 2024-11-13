# frozen_string_literal: true
module Kaminari
  module Helpers
    PARAM_KEY_EXCEPT_LIST = [:authenticity_token, :commit, :utf8, :_method, :script_name].freeze

    class Tag
      def initialize(template, params: {}, param_name: nil, theme: nil, views_prefix: nil, **options) #:nodoc:
        @template, @theme, @views_prefix, @options = template, theme, views_prefix, options
        @param_name = param_name || Kaminari.config.param_name
        @params = CGI.unescape(template.params).then { |p| p.respond_to?(:to_unsafe_h) ? p.to_unsafe_h : p }
        @params = @params.with_indifferent_access
        @params.except!(*PARAM_KEY_EXCEPT_LIST)
        @params.merge!(CGI.unescape(params))
      end
    end

    class PrevPage < Tag
      # No change in the method, as it's not using any user input directly for URL generation
    end

    class NextPage < Tag
      # No change in the method, as it's not using any user input directly for URL generation
    end
  end
end
   