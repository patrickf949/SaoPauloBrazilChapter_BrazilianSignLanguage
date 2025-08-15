# Report Website

This report site is built with [Jekyll](https://jekyllrb.com/) and hosted on [GitHub Pages](https://pages.github.com/).

## How It Works

### Jekyll Static Site Generator
- **Markdown to HTML**: Jekyll automatically converts `.md` files in this directory into static HTML pages
- **Theme**: Uses the theme specified in `_config.yml` for consistent styling

### GitHub Pages Hosting
- **Automatic Deployment**: When you push to the main branch, GitHub Pages automatically rebuilds and deploys your site
- **No Local Setup Required**: You can edit markdown files directly on GitHub or clone, edit locally, and push
- **HTTPS**: Automatically provides SSL certificates for secure connections

## Typical Workflow (Most Users)

### 1. Edit Content
- **Locally**: Clone the repo, edit files, then push changes
- **On GitHub**: Edit `.md` files directly in the browser

### 2. Push Changes
```bash
git add .
git commit -m "Update report content"
git push origin main
```

### 3. View Live Site
- Changes appear on your GitHub Pages site within a few minutes
- No local Jekyll setup required!

## File Structure
```
docs/
├── _config.yml        # Jekyll configuration and theme
├── index.md           # Homepage content
├── assets/              # images, gifs, etc.
```


## Local Development (Optional)

If you want to preview changes locally before pushing:

### Prerequisites
- Ruby (3.2+ recommended)
- Bundler gem

### Quick Setup
```bash
# Install Ruby and bundler (Ubuntu/WSL2)
sudo apt install ruby-full
gem install bundler

# Navigate to docs directory
cd docs

# Install dependencies
bundle install

# Start local server
bundle exec jekyll serve --host 0.0.0.0
```

Visit `http://localhost:4000` to preview your site locally.

### When to Rebuild vs. Reload

**Just reload the page for:**
- Content changes in `.md` files
- Asset changes (images, CSS, JS)
- Most frontend modifications

**Rebuild and restart server for:**
- Changes to `_config.yml`
- Layout/theme modifications
- Plugin changes
- Gemfile updates
- When the server seems confused

```bash
# Rebuild the site (only needed after config/layout changes)
bundle exec jekyll build

# Restart the server
bundle exec jekyll serve --host 0.0.0.0
```

**Note:** `bundle exec jekyll serve` automatically builds the site the first time and then does incremental builds with auto-regeneration. However, you still need `bundle exec jekyll build` when:
- Configuration changes aren't detected properly
- Layout/theme changes need a clean rebuild
- The incremental build gets confused
- You want to ensure a complete rebuild from scratch

### Note on Gem Environment
If you get "gems not found" errors when running `bundle exec jekyll serve`, you may need to configure user-specific gem installation:

```bash
# Set gem environment for current session
export GEM_HOME="$HOME/.gem"
export PATH="$HOME/.gem/bin:$PATH"

# Make permanent (add to ~/.bashrc)
echo 'export GEM_HOME="$HOME/.gem"' >> ~/.bashrc
echo 'export PATH="$HOME/.gem/bin:$PATH"' >> ~/.bashrc
```

Setting these environment variables will allow your local environment to find the installed gems.

**This is only needed for local development - GitHub Pages handles everything automatically when you push changes.**

## Troubleshooting

- **Changes not appearing**: Wait a few minutes for GitHub Pages to rebuild
- **Theme not working**: Check that the theme is correctly specified in `_config.yml`
- **Local preview issues**: Use `--no-watch` flag on Windows/WSL2 if auto-regeneration doesn't work
