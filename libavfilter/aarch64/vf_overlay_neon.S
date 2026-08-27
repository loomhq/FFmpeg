#include "libavutil/aarch64/asm.S"

function ff_overlay_row_44_neon, export=1
        and             w8, w4, #0xfffffff0
        mov             w7, w8
        mov             x9, x0
        cbz             w7, 99f

        movi            v30.8h, #255
        movi            v31.8h, #128
1:
        ld1             {v0.16b}, [x2], #16
        ld1             {v1.16b}, [x3], #16
        ld1             {v2.16b}, [x9]

        uxtl            v3.8h,  v0.8b
        uxtl2           v4.8h,  v0.16b
        uxtl            v5.8h,  v2.8b
        uxtl2           v6.8h,  v2.16b
        uxtl            v7.8h,  v1.8b
        uxtl2           v16.8h, v1.16b

        sub             v17.8h, v30.8h, v7.8h
        sub             v18.8h, v30.8h, v16.8h

        mul             v3.8h,  v3.8h,  v7.8h
        mul             v4.8h,  v4.8h,  v16.8h
        mla             v3.8h,  v5.8h,  v17.8h
        mla             v4.8h,  v6.8h,  v18.8h

        add             v3.8h,  v3.8h,  v31.8h
        add             v4.8h,  v4.8h,  v31.8h
        ushr            v5.8h,  v3.8h,  #8
        ushr            v6.8h,  v4.8h,  #8
        add             v3.8h,  v3.8h,  v5.8h
        add             v4.8h,  v4.8h,  v6.8h
        ushr            v3.8h,  v3.8h,  #8
        ushr            v4.8h,  v4.8h,  #8

        xtn             v0.8b,  v3.8h
        xtn2            v0.16b, v4.8h
        st1             {v0.16b}, [x9], #16

        subs            w7, w7, #16
        b.ne            1b
99:
        mov             w0, w8
        ret
endfunc

function ff_overlay_row_22_neon, export=1
        mov             w8, wzr
        cmp             w4, #1
        ble             99f

        sub             w8, w4, #1
        and             w8, w8, #0xfffffff0
        mov             w7, w8
        mov             x9, x0
        cbz             w7, 99f

        movi            v30.8h, #255
        movi            v31.8h, #128
1:
        ld1             {v0.16b}, [x2], #16
        ld1             {v1.16b}, [x9]
        ld1             {v2.16b, v3.16b}, [x3], #32

        uzp1            v4.16b, v2.16b, v3.16b
        uzp2            v5.16b, v2.16b, v3.16b

        uxtl            v6.8h,  v4.8b
        uxtl2           v7.8h,  v4.16b
        uxtl            v16.8h, v5.8b
        uxtl2           v17.8h, v5.16b
        add             v6.8h,  v6.8h,  v16.8h
        add             v7.8h,  v7.8h,  v17.8h
        ushr            v6.8h,  v6.8h,  #1
        ushr            v7.8h,  v7.8h,  #1

        uxtl            v16.8h, v4.8b
        uxtl2           v17.8h, v4.16b
        add             v6.8h,  v6.8h,  v16.8h
        add             v7.8h,  v7.8h,  v17.8h
        ushr            v6.8h,  v6.8h,  #1
        ushr            v7.8h,  v7.8h,  #1

        uxtl            v3.8h,  v0.8b
        uxtl2           v4.8h,  v0.16b
        uxtl            v5.8h,  v1.8b
        uxtl2           v16.8h, v1.16b

        sub             v17.8h, v30.8h, v6.8h
        sub             v18.8h, v30.8h, v7.8h
        mul             v3.8h,  v3.8h,  v6.8h
        mul             v4.8h,  v4.8h,  v7.8h
        mla             v3.8h,  v5.8h,  v17.8h
        mla             v4.8h,  v16.8h, v18.8h

        add             v3.8h,  v3.8h,  v31.8h
        add             v4.8h,  v4.8h,  v31.8h
        ushr            v5.8h,  v3.8h,  #8
        ushr            v16.8h, v4.8h,  #8
        add             v3.8h,  v3.8h,  v5.8h
        add             v4.8h,  v4.8h,  v16.8h
        ushr            v3.8h,  v3.8h,  #8
        ushr            v4.8h,  v4.8h,  #8

        xtn             v0.8b,  v3.8h
        xtn2            v0.16b, v4.8h
        st1             {v0.16b}, [x9], #16

        subs            w7, w7, #16
        b.ne            1b
99:
        mov             w0, w8
        ret
endfunc

function ff_overlay_row_20_neon, export=1
        mov             w8, wzr
        cmp             w4, #1
        ble             99f

        sub             w8, w4, #1
        and             w8, w8, #0xfffffff0
        mov             w7, w8
        mov             x9, x0
        cbz             w7, 99f

        add             x10, x3, x5
        movi            v30.8h, #255
        movi            v31.8h, #128
1:
        ld1             {v0.16b}, [x2], #16
        ld1             {v1.16b}, [x9]
        ld1             {v2.16b, v3.16b}, [x3], #32
        ld1             {v4.16b, v5.16b}, [x10], #32

        uzp1            v6.16b,  v2.16b, v3.16b
        uzp2            v7.16b,  v2.16b, v3.16b
        uzp1            v16.16b, v4.16b, v5.16b
        uzp2            v17.16b, v4.16b, v5.16b

        uxtl            v18.8h, v6.8b
        uxtl2           v19.8h, v6.16b
        uxtl            v20.8h, v7.8b
        uxtl2           v21.8h, v7.16b
        uxtl            v22.8h, v16.8b
        uxtl2           v23.8h, v16.16b
        uxtl            v24.8h, v17.8b
        uxtl2           v25.8h, v17.16b

        add             v18.8h, v18.8h, v20.8h
        add             v19.8h, v19.8h, v21.8h
        add             v22.8h, v22.8h, v24.8h
        add             v23.8h, v23.8h, v25.8h
        add             v18.8h, v18.8h, v22.8h
        add             v19.8h, v19.8h, v23.8h
        ushr            v18.8h, v18.8h, #2
        ushr            v19.8h, v19.8h, #2

        uxtl            v3.8h,  v0.8b
        uxtl2           v4.8h,  v0.16b
        uxtl            v5.8h,  v1.8b
        uxtl2           v16.8h, v1.16b

        sub             v20.8h, v30.8h, v18.8h
        sub             v21.8h, v30.8h, v19.8h
        mul             v3.8h,  v3.8h,  v18.8h
        mul             v4.8h,  v4.8h,  v19.8h
        mla             v3.8h,  v5.8h,  v20.8h
        mla             v4.8h,  v16.8h, v21.8h

        add             v3.8h,  v3.8h,  v31.8h
        add             v4.8h,  v4.8h,  v31.8h
        ushr            v5.8h,  v3.8h,  #8
        ushr            v16.8h, v4.8h,  #8
        add             v3.8h,  v3.8h,  v5.8h
        add             v4.8h,  v4.8h,  v16.8h
        ushr            v3.8h,  v3.8h,  #8
        ushr            v4.8h,  v4.8h,  #8

        xtn             v0.8b,  v3.8h
        xtn2            v0.16b, v4.8h
        st1             {v0.16b}, [x9], #16

        subs            w7, w7, #16
        b.ne            1b
99:
        mov             w0, w8
        ret
endfunc
