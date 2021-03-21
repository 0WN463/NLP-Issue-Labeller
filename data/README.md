# Data

Each top folder are sorted by language

Within the top folders, there are these following folders:
1. raw
	* Raw data scraped from GitHub
	* CAUTION: Some issue may appear twice (very very unlikely, haven't actually seen it) under different labels.
	This is because it was easier to scrap using filtering by labels, but it did not allow for OR labels, only AND.
2. code\_text\_split
	* This is simply `raw`, but preprocessed to separate code blocks from text via simple regex
3. links\_extracted
	* This is the HTML version of code\_text\_split
	* Images are replaced with `IMAGE_TOKEN`
	* Links are replaced by their accompanied text, and extracted into their own key-pair
	* Text is removed in key-pair (Cause GitHub is flagging that the file is too big to upload)
