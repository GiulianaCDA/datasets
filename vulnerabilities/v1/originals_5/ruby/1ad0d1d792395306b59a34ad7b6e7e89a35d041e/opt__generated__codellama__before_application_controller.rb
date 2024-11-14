class ApplicationController < ActionController::Base
  include Console::Rescue

  protect_from_forgery with: :exception
end
