import toml


def read_preset(preset):
    try:
        result = toml.load(preset)
        print(f"Successfully loaded preset from: {preset}")  # Add this
        print(f"Preset contents: {result}")  # Add this
        return result
    except Exception as e:
        print("Error: cannot read preset file. ", e)
        return None
