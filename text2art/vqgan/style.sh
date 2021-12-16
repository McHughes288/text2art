#!/bin/bash -eu

SRC_DIR=$(dirname $(realpath $0))
CODE_DIR=$(realpath $SRC_DIR/../..)

prompt=
exp_name=
styles="none artstation artstation_hd unreal deviant cryengine flickr behance 4k concept movie painting oil watercolour pastel ink_wash acrylic hot_wax digital futuristic vintage sketch lines pencil child low_poly comic cartoon escher dali van_gogh picasso morris mona_lisa gurney kinkade monet fresco renaissance printmaking abstract modern impressionism pop_art cubism surrealism contemporary fantasy graffiti anime art_deco art_nouveau bauhaus futureism expressionism minimalism geometric lowbrow tattoo lego speed 1920 1970 1990"

. ./scripts/parse_options.sh || exit 1;

[ -z "$prompt" ] && echo "please provide a prompt" && exit 1

exp_name=${exp_name:-$(date +"%Y%m%d")}

for style_name in $styles; do
    case $style_name in
        none) end_of_prompt="" ;;
        artstation) end_of_prompt=" #artstation" ;;
        artstation_hd) end_of_prompt=" trending on ArtStation HD" ;;
        unreal) end_of_prompt=" | Unreal Engine" ;;
        deviant) end_of_prompt=" | deviantart" ;;
        cryengine) end_of_prompt=" | CryEngine" ;;
        flickr) end_of_prompt=" | Flickr" ;;
        behance) end_of_prompt=" | Behance HD" ;;
        4k) end_of_prompt=" | 4K 3D" ;;
        concept) end_of_prompt=" | concept art" ;;
        movie) end_of_prompt=" | movie poster" ;;
        painting) end_of_prompt=" painting" ;;
        oil) end_of_prompt=" | oil painting" ;;
        watercolour) end_of_prompt=" | watercolour" ;;
        pastel) end_of_prompt=" | pastel" ;;
        ink_wash) end_of_prompt=" | ink wash art" ;;
        acrylic) end_of_prompt=" | acrylic art" ;;
        hot_wax) end_of_prompt=" | hot wax art" ;;
        digital) end_of_prompt=" | digital illustration" ;;
        futuristic) end_of_prompt=" | futuristic art" ;;
        vintage) end_of_prompt=" | vintage photo" ;;
        sketch) end_of_prompt=" sketch" ;;
        lines) end_of_prompt=" black lines on white background" ;;
        charcoal) end_of_prompt=" charcoal drawing" ;;
        pencil) end_of_prompt=" pencil sketch" ;;
        child) end_of_prompt=" child's drawing" ;;
        low_poly) end_of_prompt=" low poly" ;;
        comic) end_of_prompt=" in a comic #marvel" ;;
        cartoon) end_of_prompt=" as a cartoon" ;;
        escher) end_of_prompt=" by M.C. Escher" ;;
        dali)  end_of_prompt=" by Salvador Dali" ;;
        van_gogh)  end_of_prompt=" by Van Gogh" ;;
        picasso)  end_of_prompt=" by Pablo Picasso" ;;
        morris) end_of_prompt=" by William Morris" ;;
        mona_lisa) end_of_prompt=" by Mona Lisa" ;;
        gurney) end_of_prompt=" by James Gurney" ;;
        kinkade) end_of_prompt=" by Thomas Kinkade" ;;
        monet) end_of_prompt=" by Claude Monet" ;;
        fresco) end_of_prompt=" | 16th Century Fresco" ;;
        renaissance) end_of_prompt=" | Renaissance painting" ;;
        printmaking) end_of_prompt=" | printmaking" ;;
        abstract) end_of_prompt=" | abstract" ;;
        modern) end_of_prompt=" | modern" ;;
        impressionism) end_of_prompt=" | impressionism" ;;
        pop_art) end_of_prompt=" | pop art" ;;
        cubism) end_of_prompt=" | cubism" ;;
        surrealism) end_of_prompt=" #surrealism" ;;
        contemporary) end_of_prompt=" #contemporary" ;;
        fantasy) end_of_prompt=" #fantasy" ;;
        graffiti) end_of_prompt=" #graffiti" ;;
        anime) end_of_prompt=" #anime" ;;
        art_deco) end_of_prompt=" | art deco" ;;
        art_nouveau) end_of_prompt=" | art nouveau" ;;
        bauhaus) end_of_prompt=" | Bauhaus" ;;
        futureism)  end_of_prompt=" #futureism" ;;
        expressionism) end_of_prompt=" #expressionism" ;;
        minimalism) end_of_prompt=" #minimalism" ;;
        geometric) end_of_prompt=" #geometric" ;;
        lowbrow) end_of_prompt=" #lowbrow" ;;
        tattoo) end_of_prompt=" #tattoo" ;;
        lego) end_of_prompt=" made of lego" ;;
        speed) end_of_prompt=" #speedpainting" ;;
        1920) end_of_prompt=" 1920s, 1925" ;;
        1970) end_of_prompt=" 1970s, 1975" ;;
        1990) end_of_prompt=" 1990, 1995" ;;
        *) end_of_prompt=" #$style_name" ;;        
    esac
    $CODE_DIR/text2art/vqgan/run.sh --WORK_DIR /exp/$(whoami)/text2art_styles/$exp_name/$style_name --prompt "$prompt$end_of_prompt" --image_name ${exp_name}_${style_name}
done

