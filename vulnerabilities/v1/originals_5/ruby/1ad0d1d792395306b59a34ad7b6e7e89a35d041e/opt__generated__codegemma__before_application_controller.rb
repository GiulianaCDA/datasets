class ApplicationController < ActionController::Base
  include Console::Rescue

  # Removed protect_from_forgery to avoid CSRF vulnerability
end
