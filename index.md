## Parallel Regular expression matching
This project implements a fast parallel regular expression matching using OpenCL. Regular expression matching on a large number of files is an embarassingly parallel operation. In this project we explore parallellism across files as well as parallellism within files. Our implementation is based on running parallel instances of a regex engine based on Thompson's NFA. This will use OpenCL to offload the regex matching to a GPU.

You can use the [editor on GitHub](https://github.com/MadhumithaSridhara/15-418-Parallel-Project/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Proposal

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/MadhumithaSridhara/15-418-Parallel-Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
