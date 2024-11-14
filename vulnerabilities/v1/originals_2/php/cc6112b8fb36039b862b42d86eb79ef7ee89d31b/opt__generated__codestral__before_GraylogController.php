<?php
// ... rest of the code above __construct function ...

public function __construct()
{
    $timezone = Config::get('graylog.timezone');
    if ($timezone && in_array($timezone, DateTimeZone::listIdentifiers())) {
        $this->timezone = new DateTimeZone($timezone);
    } else {
        // Handle invalid timezone or use a default value
        $this->timezone = null;
    }
}

public function __invoke(Request $request, GraylogApi $api)
{
    if (! $api->isConfigured()) {
        return response()->json(['error' => 'Graylog API is not configured'], 500);
    }

    $validatedData = $request->validate([
        'device' => 'required|integer|min:1',
        // Add more validation rules for other input parameters as needed
    ]);

    // Rest of the code...
}
// ... rest of the code below __invoke function ...
