module Kaminari
  module Helpers
    PARAM_KEY_EXCEPT_LIST = [:authenticity_token, :commit, :utf8, :_method, :script_name].freeze

    class Tag
      def initialize(template, params: {}, param_name: nil, theme: nil, views_prefix: nil, **options) #:nodoc:
        @template, @theme, @views_prefix, @options = template, theme, views_prefix, options
        @param_name = param_name || Kaminari.config.param_name
        @params = template.params
        # @params in Rails 5 no longer inherits from Hash
        @params = @params.to_unsafe_h if @params.respond_to?(:to_unsafe_h)
        @params = @params.with_indifferent_access
        @params.except!(*PARAM_KEY_EXCEPT_LIST)
        @params.merge! params
      end

      def to_s(locals = {}) #:nodoc:
        formats = (@template.respond_to?(:formats) ? @template.formats : Array(@template.params[:format])) + [:html]
        @template.render partial: partial_path, locals: @options.merge(locals), formats: formats
      end

      def page_url_for(page)
        params = params_for(page)
        params[:only_path] = true
        @template.url_for params
      end

      private

      def params_for(page)
        page_params = Rack::Utils.parse_nested_query("#{@param_name}=#{page}")
        page_params = @params.deep_merge(page_params)

        if !Kaminari.config.params_on_first_page && page == 1
          page_params.delete(@param_name)
        end

        page_params
      end

      def partial_path
        File.join('kaminari', self.class.name.underscore)
      end
    end

    # ... remaining classes and code
  end
end
