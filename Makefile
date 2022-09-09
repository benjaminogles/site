
LIT_SITE_MK := lit/scripts/lit-site.mk
$(LIT_SITE_MK):
	git submodule update --init lit

include $(LIT_SITE_MK)

# copy misc assets to the build directory when they change
$(TARGET_DIR)/css: $(wildcard css/*.css)
$(TARGET_DIR)/css: | $(TARGET_DIR)
	if [ -d "$@" ]; then rm -r "$@"; fi
	cp -r "$(patsubst $(TARGET_DIR)/%,%,$@)" "$@"

# default page information
PAGE_AUTHOR := Ben Ogles
PAGE_METAS := fragments/meta.html
PAGE_HEADERS := fragments/header.html

# page-specific information
$(TARGET_DIR)/index.html: PAGE_TITLE = Ben Ogles
$(TARGET_DIR)/index.html: PAGE_DESCRIPTION = Technical articles by Ben Ogles focused on software engineering and signal processing

$(TARGET_DIR)/posts/index.html: PAGE_TITLE = All Posts - Ben Ogles
$(TARGET_DIR)/posts/index.html: PAGE_DESCRIPTION = List of all technical articles by Ben Ogles

$(TARGET_DIR)/posts/dft/index.html: PAGE_TITLE = DFT From Scratch
$(TARGET_DIR)/posts/dft/index.html: PAGE_DESCRIPTION = Derivation of the discrete Fourier transform
$(TARGET_DIR)/posts/dft/index.html: PAGE_METAS += fragments/highlightcss.html
$(TARGET_DIR)/posts/dft/index.html: PAGE_FOOTERS += fragments/highlightjs.html

