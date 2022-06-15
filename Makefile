
TARGET_DIR := build
SOURCE_DIR := src
SOURCE_DIRS := $(shell find "$(SOURCE_DIR)" -type d)
TARGET_DIRS := $(patsubst $(SOURCE_DIR)%,$(TARGET_DIR)%,$(SOURCE_DIRS))

PAGE_AUTHOR := Ben Ogles
PAGE_METAS := fragments/meta.html
PAGE_HEADERS := fragments/header.html

.PHONY: all clean

all: $(addsuffix /index.html,$(TARGET_DIRS))

clean:
	-rm -r "$(TARGET_DIR)"

$(TARGET_DIR)/index.html: PAGE_TITLE = Ben Ogles
$(TARGET_DIR)/index.html: PAGE_DESCRIPTION = Technical articles by Ben Ogles focused on software engineering and signal processing

$(TARGET_DIR)/posts/index.html: PAGE_TITLE = All Posts - Ben Ogles
$(TARGET_DIR)/posts/index.html: PAGE_DESCRIPTION = List of all technical articles by Ben Ogles

$(TARGET_DIR)/posts/dft/index.html: PAGE_TITLE = DFT From Scratch
$(TARGET_DIR)/posts/dft/index.html: PAGE_DESCRIPTION = Derivation of the discrete Fourier transform
$(TARGET_DIR)/posts/dft/index.html: PAGE_METAS += fragments/highlightcss.html
$(TARGET_DIR)/posts/dft/index.html: PAGE_FOOTERS += fragments/highlightjs.html fragments/mathjax.html

define target_page_rules =
SRC_DIR := $(1)
TGT_DIR := $(patsubst $(SOURCE_DIR)%,$(TARGET_DIR)%,$(1))
SRC_FILES := $(wildcard $(1)/[0-9]*)
TGT_FILES = $(addsuffix .md,$(patsubst $(SOURCE_DIR)%,$(TARGET_DIR)%,$(wildcard $(1)/[0-9]*)))

$$(TGT_DIR):
	mkdir -p $$@

$$(TGT_DIR)/index.html $$(TGT_FILES): | $$(TGT_DIR) $(TARGET_DIR)/css $(TARGET_DIR)/js
$$(TGT_DIR)/index.html: $$(TGT_FILES)
$$(TGT_DIR)/index.html: $(wildcard fragments/*.html)

$$(TGT_DIR)/index.html:
	lit2html \
		-a "$$(PAGE_AUTHOR)" \
		-t "$$(PAGE_TITLE)" \
		-d "$$(PAGE_DESCRIPTION)" \
		$$(foreach m,$$(PAGE_METAS),-m $$(m)) \
		$$(foreach h,$$(PAGE_HEADERS),-h $$(h)) \
		$$(foreach f,$$(PAGE_FOOTERS),-f $$(f)) \
		-o $$@ $$(filter %.md,$$^)

$$(filter %.md.md,$$(TGT_FILES)): $(TARGET_DIR)/%.md: $(SOURCE_DIR)/%
	cp "$$<" "$$@"

$$(filter-out %.md.md,$$(TGT_FILES)): $(TARGET_DIR)/%.md: $(SOURCE_DIR)/%
	lit2md -o "$$@" $$<
endef

$(foreach d,$(SOURCE_DIRS),$(eval $(call target_page_rules,$(d))))

$(TARGET_DIR)/css: css/styles.css
$(TARGET_DIR)/css $(TARGET_DIR)/js: | $(TARGET_DIR)
	if [ -d "$@" ]; then rm -r "$@"; fi
	cp -r "$(patsubst $(TARGET_DIR)/%,%,$@)" "$@"

