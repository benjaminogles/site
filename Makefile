
# these variables may be given by the command line
# all other variables are only used for readability of the make file
TARGET_DIR := build
TOOLS_DIR := subbuild

# tell make to delete targets when recipes fail
.DELETE_ON_ERROR:

# specify the default target explicitly
.DEFAULT_GOAL := all

# targets that are actually commands
.PHONY: all deps clean

# set up tools we need to build the site
LIT2MD := $(TOOLS_DIR)/bin/lit2md
LIT2HTML := $(TOOLS_DIR)/bin/lit2html
MD2HTML := $(TOOLS_DIR)/bin/md2html

# md2html lives in a submodule
md4c/md2html:
	git submodule update --init md4c
	cmake -B md4c/build -S md4c -DCMAKE_INSTALL_PREFIX=$(TOOLS_DIR)

# build and install it into the tools directory
$(MD2HTML): | md4c/md2html
	make -C md4c/build install

# lit scripts use md2html
export PATH := $(TOOLS_DIR)/bin:$(PATH)
export LD_LIBRARY_PATH := $(TOOLS_DIR)/lib:$(LD_LIBRARY_PATH)
$(LIT2MD) $(LIT2HTML): | $(MD2HTML)

# lit scripts live in a submodule
lit/lit2md lit/lit2html:
	git submodule update --init lit

# link them into the tools directory too
$(LIT2MD) $(LIT2HTML): $(TOOLS_DIR)/bin/%: lit/%
	ln -s "$(abspath $<)" "$@"

# copy misc assets to the build directory when they change
$(TARGET_DIR)/css: $(wildcard css/*.css)
$(TARGET_DIR)/css: | $(TARGET_DIR)
	if [ -d "$@" ]; then rm -r "$@"; fi
	cp -r "$(patsubst $(TARGET_DIR)/%,%,$@)" "$@"

# map each source directory to its target location
SOURCE_DIR := src
SOURCE_DIRS := $(shell find "$(SOURCE_DIR)" -type d)
TARGET_DIRS := $(patsubst $(SOURCE_DIR)%,$(TARGET_DIR)%,$(SOURCE_DIRS))

# default page information
PAGE_AUTHOR := Ben Ogles
PAGE_METAS := fragments/meta.html
PAGE_HEADERS := fragments/header.html

# default goal builds all pages
all: $(addsuffix /index.html,$(TARGET_DIRS))

# page-specific information
$(TARGET_DIR)/index.html: PAGE_TITLE = Ben Ogles
$(TARGET_DIR)/index.html: PAGE_DESCRIPTION = Technical articles by Ben Ogles focused on software engineering and signal processing

$(TARGET_DIR)/posts/index.html: PAGE_TITLE = All Posts - Ben Ogles
$(TARGET_DIR)/posts/index.html: PAGE_DESCRIPTION = List of all technical articles by Ben Ogles

$(TARGET_DIR)/posts/dft/index.html: PAGE_TITLE = DFT From Scratch
$(TARGET_DIR)/posts/dft/index.html: PAGE_DESCRIPTION = Derivation of the discrete Fourier transform
$(TARGET_DIR)/posts/dft/index.html: PAGE_METAS += fragments/highlightcss.html
$(TARGET_DIR)/posts/dft/index.html: PAGE_FOOTERS += fragments/highlightjs.html

# recipe to build a single page from a directory
define target_page_rules =

# map input directory and files to target locations
SRC_DIR := $(1)
TGT_DIR := $(patsubst $(SOURCE_DIR)%,$(TARGET_DIR)%,$(1))
SRC_FILES := $(wildcard $(1)/[0-9]*)
TGT_FILES = $(addsuffix .md,$(patsubst $(SOURCE_DIR)%,$(TARGET_DIR)%,$(wildcard $(1)/[0-9]*)))

# providing the target directory is simple
$$(TGT_DIR):
	mkdir -p $$@

# lit2html is used to build a page and install it to a target location with links to assets
$$(TGT_DIR)/index.html $$(TGT_FILES): | $$(TGT_DIR) $(TARGET_DIR)/css $(LIT2HTML)

# a page depends directly on its source files and html fragments
$$(TGT_DIR)/index.html: $$(TGT_FILES)
$$(TGT_DIR)/index.html: $(wildcard fragments/*.html)

# put together all markdown and html fragments to build the page
$$(TGT_DIR)/index.html:
	$(LIT2HTML) \
		-a "$$(PAGE_AUTHOR)" \
		-t "$$(PAGE_TITLE)" \
		-d "$$(PAGE_DESCRIPTION)" \
		$$(foreach m,$$(PAGE_METAS),-m $$(m)) \
		$$(foreach h,$$(PAGE_HEADERS),-h $$(h)) \
		$$(foreach f,$$(PAGE_FOOTERS),-f $$(f)) \
		-o $$@ $$(filter %.md,$$^)

# markdown source files can be copied to their target locations
$$(filter %.md.md,$$(TGT_FILES)): $(TARGET_DIR)/%.md: $(SOURCE_DIR)/%
	cp "$$<" "$$@"

# other markdown targets are created by lit2md
$$(filter-out %.md.md,$$(TGT_FILES)): | $(LIT2MD)

$$(filter-out %.md.md,$$(TGT_FILES)): $(TARGET_DIR)/%.md: $(SOURCE_DIR)/%
	$(LIT2MD) -o "$$@" $$<

endef

# evaluate the above template for each source directory
$(foreach d,$(SOURCE_DIRS),$(eval $(call target_page_rules,$(d))))

clean:
	-rm -r "$(TARGET_DIR)" "$(TOOLS_DIR)"
	-git clean -xf $(SOURCE_DIR)
