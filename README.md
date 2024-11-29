# YouTube2Blog

A Python script to fetch subtitles from YouTube videos and generate detailed Japanese blog articles using OpenAI API. This tool automates the process of transforming video content into readable and structured blog articles.

## Features

- Fetch subtitles from YouTube videos by providing a URL and language code.
- Generate a 2000-3000 character Japanese blog article using OpenAI API.
- Save the generated blog article as a text file.

## Requirements

- Python 3.7 or later
- Python libraries:
  - `youtube-transcript-api`
  - `openai`
- OpenAI API key

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/takurot/youtube2blog.git
   cd youtube2blog
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable or specify it directly in the script:

   - To set it as an environment variable:
     ```bash
     export OPENAI_API_KEY="your_api_key_here"
     ```

## Usage

Run the script with the following command:

```bash
python youtube2blog.py "Language Code" "YouTube Video URL"
```

### Arguments

- **`Language Code`**: The language code of the subtitles (e.g., `en`, `ja`).
- **`YouTube Video URL`**: The URL of the YouTube video containing the subtitles.

### Example

To fetch Japanese subtitles and generate a blog article:

```bash
python youtube2blog.py ja https://www.youtube.com/watch?v=example_video_id
```

### Output

The generated blog article will be saved as a text file in the script’s directory with the following format:

```
blog_article_<video_id>.txt
```

Example:

```
blog_article_example_video_id.txt
```

## File Structure

```
youtube2blog/
│
├── youtube_to_blog.py     # Main script
├── requirements.txt       # Required dependencies
└── README.md              # This file
```

## Notes

- Subtitles must be available for the specified YouTube video; otherwise, transcription will fail.
- An OpenAI API key is required, and API usage fees may apply.
- For long videos, token limits may require additional logic to split the transcript into smaller parts.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
