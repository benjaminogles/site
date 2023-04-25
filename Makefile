
LIT_SITE_MK := lit/scripts/lit-site.mk
$(LIT_SITE_MK):
	git submodule update --init lit

include $(LIT_SITE_MK)

$(TARGET_DIR)/css:
	mkdir -p $@
STYLESHEETS := $(addprefix $(TARGET_DIR)/, $(wildcard css/*.css))
$(STYLESHEETS): | $(TARGET_DIR)/css
$(STYLESHEETS): $(TARGET_DIR)/%: %
	cp "$<" "$@"

all: $(STYLESHEETS)

# default page information
PAGE_AUTHOR := Ben Ogles
PAGE_METAS := fragments/meta.html
PAGE_HEADERS := fragments/header.html

# page-specific information
$(TARGET_DIR)/index.html: PAGE_TITLE = Ben Ogles
$(TARGET_DIR)/index.html: PAGE_DESCRIPTION = Technical articles by Ben Ogles focused on software engineering and signal processing

$(TARGET_DIR)/posts/index.html: PAGE_TITLE = All Posts - Ben Ogles
$(TARGET_DIR)/posts/index.html: PAGE_DESCRIPTION = List of all technical articles by Ben Ogles

$(TARGET_DIR)/posts/complex-signals/index.html: PAGE_TITLE = Complex Signals
$(TARGET_DIR)/posts/complex-signals/index.html: PAGE_DESCRIPTION = How complex numbers are used in signal processing
$(TARGET_DIR)/posts/complex-signals/index.html: PAGE_METAS += fragments/highlightcss.html
$(TARGET_DIR)/posts/complex-signals/index.html: PAGE_FOOTERS += fragments/highlightjs.html

$(TARGET_DIR)/posts/nyquist-frequency/index.html: PAGE_TITLE = Nyquist Frequency
$(TARGET_DIR)/posts/nyquist-frequency/index.html: PAGE_DESCRIPTION = The limits of representing a continuous signal by its discrete samples
$(TARGET_DIR)/posts/nyquist-frequency/index.html: PAGE_METAS += fragments/highlightcss.html
$(TARGET_DIR)/posts/nyquist-frequency/index.html: PAGE_FOOTERS += fragments/highlightjs.html

$(TARGET_DIR)/posts/dft/index.html: PAGE_TITLE = The Discrete Fourier Transform
$(TARGET_DIR)/posts/dft/index.html: PAGE_DESCRIPTION = Derivation of the discrete Fourier transform (DFT)
$(TARGET_DIR)/posts/dft/index.html: PAGE_METAS += fragments/highlightcss.html
$(TARGET_DIR)/posts/dft/index.html: PAGE_FOOTERS += fragments/highlightjs.html

$(TARGET_DIR)/posts/lti/index.html: PAGE_TITLE = Linear Time-Invariant Systems
$(TARGET_DIR)/posts/lti/index.html: PAGE_DESCRIPTION = Introduction to Linear and Time-Invariant (LTI) Systems
$(TARGET_DIR)/posts/lti/index.html: PAGE_METAS += fragments/highlightcss.html
$(TARGET_DIR)/posts/lti/index.html: PAGE_FOOTERS += fragments/highlightjs.html
