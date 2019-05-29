# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 19.0.1.144 Build 20181018";
# mark_description "--vec-report -S";
	.file "vectorize.c"
	.section	__TEXT, __text
L_TXTST0:
L_2__routine_start__main_0:
# -- Begin  _main
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _main
# --- main(int, const char **)
_main:
# parameter 1: %edi
# parameter 2: %rsi
L_B1.1:                         # Preds L_B1.0
                                # Execution count [1.00e+00]
L_LCFI1:
L____tag_value__main.1:
L_L2:
                                                          #51.39
        pushq     %rbp                                          #51.39
L_LCFI2:
        movq      %rsp, %rbp                                    #51.39
L_LCFI3:
        andq      $-128, %rsp                                   #51.39
        pushq     %r12                                          #51.39
        pushq     %r13                                          #51.39
        pushq     %r14                                          #51.39
        pushq     %r15                                          #51.39
        pushq     %rbx                                          #51.39
        subq      $216, %rsp                                    #51.39
        xorl      %esi, %esi                                    #51.39
        movl      $3, %edi                                      #51.39
        call      ___intel_new_feature_proc_init                #51.39
L_LCFI4:
                                # LOE rbx r14 r15
L_B1.33:                        # Preds L_B1.1
                                # Execution count [1.00e+00]
        stmxcsr   (%rsp)                                        #51.39
        xorl      %edi, %edi                                    #53.8
        orl       $32832, (%rsp)                                #51.39
        ldmxcsr   (%rsp)                                        #51.39
#       time(time_t *)
        call      _time                                         #53.8
                                # LOE rax rbx r14 r15
L_B1.2:                         # Preds L_B1.33
                                # Execution count [1.00e+00]
        movl      %eax, %edi                                    #53.2
#       srand(unsigned int)
        call      _srand                                        #53.2
                                # LOE rbx r14 r15
L_B1.3:                         # Preds L_B1.2
                                # Execution count [1.00e+00]
        movaps    _shape.835.0.0.4(%rip), %xmm0                 #56.21
        movslq    _shape.835.0.0.4(%rip), %rax                  #56.21
        movaps    %xmm0, 16(%rsp)                               #56.21
                                # LOE rax rbx r14 r15
L_B1.4:                         # Preds L_B1.3
                                # Execution count [1.00e+00]
        lea       (,%rax,8), %rdi                               #57.25
#       malloc(size_t)
        call      _malloc                                       #57.25
                                # LOE rax rbx r14 r15
L_B1.34:                        # Preds L_B1.4
                                # Execution count [1.00e+00]
        movq      %rax, %r13                                    #57.25
                                # LOE rbx r13 r14 r15
L_B1.5:                         # Preds L_B1.34
                                # Execution count [1.00e+00]
        movq      %r13, %rdi                                    #58.2
        lea       16(%rsp), %rsi                                #58.2
L____tag_value__main.11:
#       Initialize4D(complex128_t ****, const int *)
        call      _Initialize4D                                 #58.2
L____tag_value__main.12:
                                # LOE rbx r13 r14 r15
L_B1.6:                         # Preds L_B1.5
                                # Execution count [1.00e+00]
        movq      %r13, %rdi                                    #60.2
        lea       16(%rsp), %rsi                                #60.2
L____tag_value__main.13:
#       Print4D(complex128_t ****, const int *)
        call      _Print4D                                      #60.2
L____tag_value__main.14:
                                # LOE rbx r13 r14 r15
L_B1.7:                         # Preds L_B1.6
                                # Execution count [1.00e+00]
        movl      16(%rsp), %r9d                                #66.30
        movl      %r9d, %ecx                                    #66.21
        movl      24(%rsp), %esi                                #66.30
        movl      %esi, %edi                                    #66.21
        movl      20(%rsp), %r8d                                #66.30
        movl      28(%rsp), %r11d                               #66.30
        xorps     %xmm13, %xmm13                                #66.21
        imull     %r11d, %edi                                   #66.21
        imull     %r8d, %ecx                                    #66.21
        imull     %edi, %ecx                                    #66.21
        testl     %ecx, %ecx                                    #66.21
        jle       L_B1.16       # Prob 50%                      #66.21
                                # LOE rbx r13 r14 r15 ecx esi edi r8d r9d r11d xmm13
L_B1.8:                         # Preds L_B1.7
                                # Execution count [9.00e-01]
        cmpl      $4, %ecx                                      #66.21
        jl        L_B1.29       # Prob 10%                      #66.21
                                # LOE rbx r13 r14 r15 ecx esi edi r8d r9d r11d xmm13
L_B1.9:                         # Preds L_B1.8
                                # Execution count [9.00e-01]
        movl      %r8d, %edx                                    #66.21
        movd      %edi, %xmm3                                   #66.21
        imull     %esi, %edx                                    #66.21
        movd      %r9d, %xmm1                                   #66.30
        movd      %r8d, %xmm5                                   #66.30
        movd      %r11d, %xmm6                                  #66.30
        movd      %esi, %xmm7                                   #66.30
        movaps    %xmm13, %xmm14                                #66.21
        movhpd    %xmm13, 8(%rsp)                               #66.21
        movaps    %xmm13, %xmm12                                #66.21
        pshufd    $0, %xmm3, %xmm3                              #66.21
        movaps    %xmm12, %xmm11                                #66.21
        imull     %r11d, %edx                                   #66.21
        movl      %ecx, %r12d                                   #66.21
        pshufd    $0, %xmm1, %xmm1                              #66.30
        pshufd    $0, %xmm5, %xmm15                             #66.30
        pshufd    $0, %xmm6, %xmm6                              #66.30
        pshufd    $0, %xmm7, %xmm5                              #66.30
        xorl      %eax, %eax                                    #66.21
        andl      $-4, %r12d                                    #66.21
        movsd     %xmm13, (%rsp)                                #66.21
        movl      %eax, %r14d                                   #66.30
        movhpd    8(%rsp), %xmm14                               #66.21
        movl      %ecx, %ebx                                    #66.30
        movd      %edx, %xmm2                                   #66.21
        movl      %edi, %r15d                                   #66.30
        pshufd    $0, %xmm2, %xmm2                              #66.21
        movdqa    L_2il0floatpacket.1(%rip), %xmm8              #66.21
        movdqa    %xmm5, 64(%rsp)                               #66.30[spill]
        movdqa    %xmm6, 144(%rsp)                              #66.30[spill]
        movdqa    %xmm15, 80(%rsp)                              #66.30[spill]
        movdqa    %xmm1, 96(%rsp)                               #66.30[spill]
        movdqa    %xmm2, 112(%rsp)                              #66.30[spill]
        movdqa    %xmm3, 128(%rsp)                              #66.30[spill]
        movl      %r11d, 32(%rsp)                               #66.30[spill]
        movl      %esi, 40(%rsp)                                #66.30[spill]
        movl      %r8d, 48(%rsp)                                #66.30[spill]
        movl      %r9d, 56(%rsp)                                #66.30[spill]
                                # LOE r13 ebx r12d r14d r15d xmm8 xmm11 xmm12 xmm13 xmm14
L_B1.10:                        # Preds L_B1.35 L_B1.9
                                # Execution count [5.00e+00]
        movdqa    112(%rsp), %xmm1                              #66.21[spill]
        movdqa    %xmm8, %xmm0                                  #66.21
        call      ___svml_idiv4                                 #66.21
                                # LOE r13 ebx r12d r14d r15d xmm0 xmm8 xmm11 xmm12 xmm13 xmm14
L_B1.41:                        # Preds L_B1.10
                                # Execution count [5.00e+00]
        movdqa    96(%rsp), %xmm1                               #66.21[spill]
        call      ___svml_irem4                                 #66.21
                                # LOE r13 ebx r12d r14d r15d xmm0 xmm8 xmm11 xmm12 xmm13 xmm14
L_B1.40:                        # Preds L_B1.41
                                # Execution count [5.00e+00]
        movd      %xmm0, %rsi                                   #66.21
        psrldq    $8, %xmm0                                     #66.21
        movdqa    128(%rsp), %xmm1                              #66.21[spill]
        movd      %xmm0, %r9                                    #66.21
        movdqa    %xmm8, %xmm0                                  #66.21
        movslq    %esi, %rdx                                    #66.21
        movslq    %r9d, %r8                                     #66.21
        sarq      $32, %rsi                                     #66.21
        sarq      $32, %r9                                      #66.21
        movq      (%r13,%rdx,8), %xmm9                          #66.21
        movq      (%r13,%r8,8), %xmm15                          #66.21
        movhpd    (%r13,%rsi,8), %xmm9                          #66.21
        movhpd    (%r13,%r9,8), %xmm15                          #66.21
        call      ___svml_idiv4                                 #66.21
                                # LOE r13 ebx r12d r14d r15d xmm0 xmm8 xmm9 xmm11 xmm12 xmm13 xmm14 xmm15
L_B1.39:                        # Preds L_B1.40
                                # Execution count [5.00e+00]
        movdqa    80(%rsp), %xmm1                               #66.21[spill]
        call      ___svml_irem4                                 #66.21
                                # LOE r13 ebx r12d r14d r15d xmm0 xmm8 xmm9 xmm11 xmm12 xmm13 xmm14 xmm15
L_B1.38:                        # Preds L_B1.39
                                # Execution count [5.00e+00]
        movdqa    %xmm0, %xmm10                                 #66.21
        punpckldq %xmm0, %xmm10                                 #66.21
        psrldq    $8, %xmm0                                     #66.21
        movdqa    %xmm10, %xmm2                                 #66.21
        punpckldq %xmm0, %xmm0                                  #66.21
        psrad     $31, %xmm2                                    #66.21
        movdqa    %xmm0, %xmm5                                  #66.21
        movdqa    L_2il0floatpacket.2(%rip), %xmm3              #66.21
        psrad     $31, %xmm5                                    #66.21
        movdqa    L_2il0floatpacket.3(%rip), %xmm4              #66.21
        pand      %xmm3, %xmm2                                  #66.21
        pand      %xmm4, %xmm10                                 #66.21
        pand      %xmm3, %xmm5                                  #66.21
        pand      %xmm4, %xmm0                                  #66.21
        por       %xmm10, %xmm2                                 #66.21
        por       %xmm0, %xmm5                                  #66.21
        psllq     $3, %xmm2                                     #66.21
        psllq     $3, %xmm5                                     #66.21
        movdqa    %xmm8, %xmm0                                  #66.21
        paddq     %xmm2, %xmm9                                  #66.21
        paddq     %xmm5, %xmm15                                 #66.21
        movd      %xmm9, %rdx                                   #66.21
        punpckhqdq %xmm9, %xmm9                                 #66.21
        movdqa    144(%rsp), %xmm1                              #66.21[spill]
        movd      %xmm15, %r8                                   #66.21
        punpckhqdq %xmm15, %xmm15                               #66.21
        movq      (%rdx), %xmm10                                #66.21
        movd      %xmm9, %rsi                                   #66.21
        movd      %xmm15, %r9                                   #66.21
        movq      (%r8), %xmm9                                  #66.21
        movhpd    (%rsi), %xmm10                                #66.21
        movhpd    (%r9), %xmm9                                  #66.21
        call      ___svml_idiv4                                 #66.21
                                # LOE r13 ebx r12d r14d r15d xmm0 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14
L_B1.37:                        # Preds L_B1.38
                                # Execution count [5.00e+00]
        movdqa    64(%rsp), %xmm1                               #66.21[spill]
        call      ___svml_irem4                                 #66.21
                                # LOE r13 ebx r12d r14d r15d xmm0 xmm8 xmm9 xmm10 xmm11 xmm12 xmm13 xmm14
L_B1.36:                        # Preds L_B1.37
                                # Execution count [5.00e+00]
        movdqa    %xmm0, %xmm15                                 #66.21
        punpckldq %xmm0, %xmm15                                 #66.21
        psrldq    $8, %xmm0                                     #66.21
        movdqa    %xmm15, %xmm2                                 #66.21
        punpckldq %xmm0, %xmm0                                  #66.21
        psrad     $31, %xmm2                                    #66.21
        movdqa    %xmm0, %xmm5                                  #66.21
        movdqa    L_2il0floatpacket.2(%rip), %xmm3              #66.21
        psrad     $31, %xmm5                                    #66.21
        movdqa    L_2il0floatpacket.3(%rip), %xmm4              #66.21
        pand      %xmm3, %xmm2                                  #66.21
        pand      %xmm4, %xmm15                                 #66.21
        pand      %xmm3, %xmm5                                  #66.21
        pand      %xmm4, %xmm0                                  #66.21
        por       %xmm15, %xmm2                                 #66.21
        por       %xmm0, %xmm5                                  #66.21
        psllq     $3, %xmm2                                     #66.21
        psllq     $3, %xmm5                                     #66.21
        movdqa    %xmm8, %xmm0                                  #66.21
        paddq     %xmm2, %xmm10                                 #66.21
        paddq     %xmm5, %xmm9                                  #66.21
        movd      %xmm10, %rdx                                  #66.21
        punpckhqdq %xmm10, %xmm10                               #66.21
        movdqa    144(%rsp), %xmm1                              #66.21[spill]
        movd      %xmm9, %r8                                    #66.21
        punpckhqdq %xmm9, %xmm9                                 #66.21
        movq      (%rdx), %xmm15                                #66.21
        movd      %xmm10, %rsi                                  #66.21
        movd      %xmm9, %r9                                    #66.21
        movq      (%r8), %xmm10                                 #66.21
        movhpd    (%rsi), %xmm15                                #66.21
        movhpd    (%r9), %xmm10                                 #66.21
        call      ___svml_irem4                                 #66.21
                                # LOE r13 ebx r12d r14d r15d xmm0 xmm8 xmm10 xmm11 xmm12 xmm13 xmm14 xmm15
L_B1.35:                        # Preds L_B1.36
                                # Execution count [5.00e+00]
        movdqa    %xmm0, %xmm1                                  #66.21
        addl      $4, %r14d                                     #66.21
        punpckldq %xmm0, %xmm1                                  #66.21
        cmpl      %r12d, %r14d                                  #66.21
        movdqa    %xmm1, %xmm2                                  #66.21
        movdqa    L_2il0floatpacket.2(%rip), %xmm3              #66.21
        psrad     $31, %xmm2                                    #66.21
        movdqa    L_2il0floatpacket.3(%rip), %xmm4              #66.21
        pand      %xmm3, %xmm2                                  #66.21
        psrldq    $8, %xmm0                                     #66.21
        pand      %xmm4, %xmm1                                  #66.21
        punpckldq %xmm0, %xmm0                                  #66.21
        por       %xmm1, %xmm2                                  #66.21
        movdqa    %xmm0, %xmm5                                  #66.21
        psllq     $4, %xmm2                                     #66.21
        psrad     $31, %xmm5                                    #66.21
        pand      %xmm4, %xmm0                                  #66.21
        paddq     %xmm2, %xmm15                                 #66.21
        pand      %xmm3, %xmm5                                  #66.21
        por       %xmm0, %xmm5                                  #66.21
        movd      %xmm15, %rdx                                  #66.21
        psllq     $4, %xmm5                                     #66.21
        paddq     %xmm5, %xmm10                                 #66.21
        punpckhqdq %xmm15, %xmm15                               #66.21
        movq      (%rdx), %rsi                                  #66.21
        movq      8(%rdx), %r8                                  #66.21
        movd      %xmm10, %rdx                                  #66.21
        movd      %rsi, %xmm9                                   #66.21
        punpckhqdq %xmm10, %xmm10                               #66.21
        movd      %r8, %xmm6                                    #66.21
        movq      (%rdx), %rsi                                  #66.21
        movq      8(%rdx), %rdx                                 #66.21
        movd      %xmm15, %r9                                   #66.21
        movd      %rsi, %xmm2                                   #66.21
        movd      %rdx, %xmm7                                   #66.21
        unpcklpd  %xmm6, %xmm9                                  #66.21
        movd      %xmm10, %rdx                                  #66.21
        unpcklpd  %xmm7, %xmm2                                  #66.21
        paddd     L_2il0floatpacket.0(%rip), %xmm8              #66.21
        movq      (%rdx), %rsi                                  #66.21
        movq      8(%rdx), %rdx                                 #66.21
        movq      (%r9), %r10                                   #66.21
        movq      8(%r9), %r11                                  #66.21
        movd      %rsi, %xmm3                                   #66.21
        movd      %rdx, %xmm10                                  #66.21
        movd      %r10, %xmm1                                   #66.21
        movd      %r11, %xmm15                                  #66.21
        unpcklpd  %xmm15, %xmm1                                 #66.21
        unpcklpd  %xmm10, %xmm3                                 #66.21
        addpd     %xmm9, %xmm14                                 #66.21
        addpd     %xmm1, %xmm13                                 #66.21
        addpd     %xmm2, %xmm12                                 #66.21
        addpd     %xmm3, %xmm11                                 #66.21
        jb        L_B1.10       # Prob 82%                      #66.21
                                # LOE r13 ebx r12d r14d r15d xmm8 xmm11 xmm12 xmm13 xmm14
L_B1.11:                        # Preds L_B1.35
                                # Execution count [9.00e-01]
        addpd     %xmm13, %xmm14                                #66.21
        addpd     %xmm12, %xmm14                                #66.21
        addpd     %xmm11, %xmm14                                #66.21
        movaps    %xmm14, (%rsp)                                #66.21
        movl      %ebx, %ecx                                    #
        movsd     (%rsp), %xmm13                                #66.21
        movl      %r15d, %edi                                   #
        movl      32(%rsp), %r11d                               #[spill]
        movl      40(%rsp), %esi                                #[spill]
        movl      48(%rsp), %r8d                                #[spill]
        movl      56(%rsp), %r9d                                #[spill]
        movhpd    8(%rsp), %xmm13                               #66.21
                                # LOE rbx r13 r14 r15 ecx esi edi r8d r9d r11d r12d xmm13
L_B1.12:                        # Preds L_B1.11 L_B1.29
                                # Execution count [1.00e+00]
        cmpl      %ecx, %r12d                                   #66.21
        jae       L_B1.16       # Prob 9%                       #66.21
                                # LOE rbx r13 r14 r15 ecx esi edi r8d r9d r11d r12d xmm13
L_B1.13:                        # Preds L_B1.12
                                # Execution count [9.00e-01]
        movl      %r8d, %r10d                                   #66.21
        imull     %esi, %r10d                                   #66.21
        imull     %r11d, %r10d                                  #66.21
        .align    4
                                # LOE r13 ecx esi edi r8d r9d r10d r11d r12d xmm13
L_B1.14:                        # Preds L_B1.14 L_B1.13
                                # Execution count [5.00e+00]
        movl      %r12d, %eax                                   #66.21
        cltd                                                    #66.21
        idivl     %r11d                                         #66.21
        movslq    %edx, %r15                                    #66.21
        cltd                                                    #66.21
        idivl     %esi                                          #66.21
        movl      %r12d, %eax                                   #66.21
        movslq    %edx, %rbx                                    #66.21
        cltd                                                    #66.21
        idivl     %edi                                          #66.21
        shlq      $4, %r15                                      #66.21
        cltd                                                    #66.21
        idivl     %r8d                                          #66.21
        movl      %r12d, %eax                                   #66.21
        incl      %r12d                                         #66.21
        movslq    %edx, %r14                                    #66.21
        cltd                                                    #66.21
        idivl     %r10d                                         #66.21
        cltd                                                    #66.21
        idivl     %r9d                                          #66.21
        cmpl      %ecx, %r12d                                   #66.21
        movslq    %edx, %rdx                                    #66.21
        movq      (%r13,%rdx,8), %rax                           #66.21
        movq      (%rax,%r14,8), %r14                           #66.21
        movq      (%r14,%rbx,8), %rax                           #66.21
        movsd     (%rax,%r15), %xmm0                            #66.21
        movhpd    8(%rax,%r15), %xmm0                           #66.21
        addpd     %xmm0, %xmm13                                 #66.21
        jb        L_B1.14       # Prob 82%                      #66.21
                                # LOE r13 ecx esi edi r8d r9d r10d r11d r12d xmm13
L_B1.16:                        # Preds L_B1.7 L_B1.14 L_B1.12
                                # Execution count [1.00e+00]
        movaps    %xmm13, %xmm0                                 #67.2
        lea       L_2__STRING.0(%rip), %rdi                     #67.2
        unpckhpd  %xmm13, %xmm13                                #67.2
        movl      $2, %eax                                      #67.2
        movaps    %xmm13, %xmm1                                 #67.2
L____tag_value__main.36:
#       printf(const char *, ...)
        call      _printf                                       #67.2
L____tag_value__main.37:
                                # LOE rbx r13 r14 r15
L_B1.17:                        # Preds L_B1.16
                                # Execution count [1.00e+00]
        movslq    16(%rsp), %rdi                                #70.30
        shlq      $3, %rdi                                      #70.30
#       malloc(size_t)
        call      _malloc                                       #70.30
                                # LOE rax rbx r13 r14 r15
L_B1.42:                        # Preds L_B1.17
                                # Execution count [1.00e+00]
        movq      %rax, %r12                                    #70.30
                                # LOE rbx r12 r13 r14 r15
L_B1.18:                        # Preds L_B1.42
                                # Execution count [1.00e+00]
        movq      %r12, %rdi                                    #71.2
        lea       16(%rsp), %rsi                                #71.2
L____tag_value__main.38:
#       Initialize2D(complex128_t **, const int *)
        call      _Initialize2D                                 #71.2
L____tag_value__main.39:
                                # LOE rbx r12 r13 r14 r15
L_B1.19:                        # Preds L_B1.18
                                # Execution count [1.00e+00]
        movl      16(%rsp), %r11d                               #72.16
        movl      %r11d, %edx                                   #72.2
        movl      24(%rsp), %r9d                                #72.16
        movl      %r9d, %esi                                    #72.2
        movl      20(%rsp), %r10d                               #72.16
        xorl      %ecx, %ecx                                    #72.2
        movl      28(%rsp), %r8d                                #72.16
        imull     %r8d, %esi                                    #72.2
        imull     %r10d, %edx                                   #72.2
        imull     %esi, %edx                                    #72.2
        testl     %edx, %edx                                    #72.2
        jle       L_B1.23       # Prob 10%                      #72.2
                                # LOE rbx r12 r13 r14 r15 edx ecx esi r8d r9d r10d r11d
L_B1.20:                        # Preds L_B1.19
                                # Execution count [9.00e-01]
        movl      %r10d, %eax                                   #72.2
        imull     %r9d, %eax                                    #72.2
        imull     %r8d, %eax                                    #72.2
        movl      %eax, 32(%rsp)                                #72.2[spill]
        movl      %edx, (%rsp)                                  #72.2[spill]
        movq      %r13, 8(%rsp)                                 #72.2[spill]
        .align    4
                                # LOE r12 ecx esi r8d r9d r10d r11d
L_B1.21:                        # Preds L_B1.21 L_B1.20
                                # Execution count [5.00e+00]
        movl      %ecx, %eax                                    #72.2
        cltd                                                    #72.2
        idivl     %esi                                          #72.2
        cltd                                                    #72.2
        idivl     %r10d                                         #72.2
        movl      %ecx, %eax                                    #72.2
        movslq    %edx, %r13                                    #72.2
        cltd                                                    #72.2
        idivl     32(%rsp)                                      #72.2[spill]
        movq      %r13, %r15                                    #72.2
        cltd                                                    #72.2
        idivl     %r11d                                         #72.2
        movl      %ecx, %eax                                    #72.2
        incl      %ecx                                          #72.2
        movslq    %edx, %rbx                                    #72.2
        cltd                                                    #72.2
        idivl     %r8d                                          #72.2
        movslq    %edx, %r14                                    #72.2
        cltd                                                    #72.2
        idivl     %r9d                                          #72.2
        movq      8(%rsp), %rax                                 #72.2[spill]
        movq      (%r12,%rbx,8), %rdi                           #72.2
        movslq    %edx, %rdx                                    #72.2
        movq      (%rax,%rbx,8), %rbx                           #72.2
        shlq      $4, %r15                                      #72.2
        shlq      $4, %r14                                      #72.2
        movq      (%rbx,%r13,8), %r13                           #72.2
        cmpl      (%rsp), %ecx                                  #72.2[spill]
        movsd     (%r15,%rdi), %xmm1                            #72.2
        movq      (%r13,%rdx,8), %rax                           #72.2
        movhpd    8(%r15,%rdi), %xmm1                           #72.2
        movsd     (%rax,%r14), %xmm0                            #72.2
        movhpd    8(%rax,%r14), %xmm0                           #72.2
        addpd     %xmm0, %xmm1                                  #72.2
        movsd     %xmm1, (%r15,%rdi)                            #72.2
        movhpd    %xmm1, 8(%r15,%rdi)                           #72.2
        jl        L_B1.21       # Prob 82%                      #72.2
                                # LOE r12 ecx esi r8d r9d r10d r11d
L_B1.22:                        # Preds L_B1.21
                                # Execution count [9.00e-01]
        movq      8(%rsp), %r13                                 #[spill]
                                # LOE rbx r12 r13 r14 r15
L_B1.23:                        # Preds L_B1.22 L_B1.19
                                # Execution count [1.00e+00]
        movq      %r12, %rdi                                    #73.2
        lea       16(%rsp), %rsi                                #73.2
L____tag_value__main.47:
#       Print2D(complex128_t **, const int *)
        call      _Print2D                                      #73.2
L____tag_value__main.48:
                                # LOE rbx r12 r13 r14 r15
L_B1.24:                        # Preds L_B1.23
                                # Execution count [1.00e+00]
        movq      %r13, %rdi                                    #76.2
        lea       16(%rsp), %rsi                                #76.2
L____tag_value__main.49:
#       Free4D(complex128_t ****, const int *)
        call      _Free4D                                       #76.2
L____tag_value__main.50:
                                # LOE rbx r12 r13 r14 r15
L_B1.25:                        # Preds L_B1.24
                                # Execution count [1.00e+00]
        movq      %r13, %rdi                                    #77.2
#       free(void *)
        call      _free                                         #77.2
                                # LOE rbx r12 r14 r15
L_B1.26:                        # Preds L_B1.25
                                # Execution count [1.00e+00]
        movq      %r12, %rdi                                    #78.2
        lea       16(%rsp), %rsi                                #78.2
L____tag_value__main.51:
#       Free2D(complex128_t **, const int *)
        call      _Free2D                                       #78.2
L____tag_value__main.52:
                                # LOE rbx r12 r14 r15
L_B1.27:                        # Preds L_B1.26
                                # Execution count [1.00e+00]
        movq      %r12, %rdi                                    #79.2
#       free(void *)
        call      _free                                         #79.2
                                # LOE rbx r14 r15
L_B1.28:                        # Preds L_B1.27
                                # Execution count [1.00e+00]
        xorl      %eax, %eax                                    #80.9
        addq      $216, %rsp                                    #80.9
L_LCFI5:
        popq      %rbx                                          #80.9
L_LCFI6:
        popq      %r15                                          #80.9
L_LCFI7:
        popq      %r14                                          #80.9
L_LCFI8:
        popq      %r13                                          #80.9
L_LCFI9:
        popq      %r12                                          #80.9
        movq      %rbp, %rsp                                    #80.9
        popq      %rbp                                          #80.9
L_LCFI10:
        ret                                                     #80.9
L_LCFI11:
                                # LOE
L_B1.29:                        # Preds L_B1.8
                                # Execution count [9.00e-02]: Infreq
        xorl      %r12d, %r12d                                  #66.21
        jmp       L_B1.12       # Prob 100%                     #66.21
        .align    4
                                # LOE rbx r13 r14 r15 ecx esi edi r8d r9d r11d r12d xmm13
L_LCFI12:
# mark_end;
L..LN_main.0:
	.section	__TEXT, __const
	.align 4
	.align 4
_shape.835.0.0.4:
	.long	4
	.long	4
	.long	4
	.long	4
	.section	__DATA, __data
# -- End  _main
	.section	__TEXT, __text
L_2__routine_start__Sum_1:
# -- Begin  _Sum
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Sum
# --- Sum(complex128_t ****, const int *)
_Sum:
# parameter 1: %rdi
# parameter 2: %rsi
L_B2.1:                         # Preds L_B2.0
                                # Execution count [1.00e+00]
L_LCFI13:
L____tag_value__Sum.68:
L_L69:
                                                         #11.59
        pushq     %r12                                          #11.59
L_LCFI14:
        pushq     %r13                                          #11.59
L_LCFI15:
        pushq     %r14                                          #11.59
L_LCFI16:
        pushq     %r15                                          #11.59
L_LCFI17:
        pushq     %rbx                                          #11.59
L_LCFI18:
        subq      $160, %rsp                                    #11.59
L_LCFI19:
        movq      %rsi, %rax                                    #11.59
        movq      %rdi, %rdx                                    #11.59
        xorps     %xmm13, %xmm13                                #15.19
        movl      4(%rax), %edi                                 #14.30
        movl      %edi, %r12d                                   #14.30
        movl      12(%rax), %r10d                               #14.52
        movl      %r10d, %r11d                                  #14.52
        movl      (%rax), %r8d                                  #14.19
        movl      8(%rax), %esi                                 #14.41
        imull     %esi, %r12d                                   #14.30
        imull     %r8d, %r11d                                   #14.52
        imull     %r12d, %r11d                                  #14.41
        testl     %r11d, %r11d                                  #16.18
        jle       L_B2.10       # Prob 50%                      #16.18
                                # LOE rdx rbx rbp r13 r14 r15 esi edi r8d r10d r11d r12d xmm13
L_B2.2:                         # Preds L_B2.1
                                # Execution count [9.00e-01]
        cmpl      $4, %r11d                                     #16.2
        jl        L_B2.11       # Prob 10%                      #16.2
                                # LOE rdx rbx rbp r13 r14 r15 esi edi r8d r10d r11d r12d xmm13
L_B2.3:                         # Preds L_B2.2
                                # Execution count [9.00e-01]
        movl      %r10d, %r9d                                   #20.40
        movl      %r11d, %ecx                                   #16.2
        imull     %r12d, %r9d                                   #20.40
        movd      %r8d, %xmm2                                   #20.54
        movd      %edi, %xmm7                                   #19.43
        movd      %r10d, %xmm8                                  #17.11
        movd      %esi, %xmm10                                  #18.30
        movaps    %xmm13, %xmm15                                #15.19
        movhpd    %xmm13, 8(%rsp)                               #15.19
        movaps    %xmm13, %xmm6                                 #15.19
        pshufd    $0, %xmm2, %xmm2                              #20.54
        movaps    %xmm6, %xmm4                                  #15.19
        pshufd    $0, %xmm7, %xmm14                             #19.43
        andl      $-4, %ecx                                     #16.2
        pshufd    $0, %xmm8, %xmm8                              #17.11
        xorl      %eax, %eax                                    #16.2
        pshufd    $0, %xmm10, %xmm7                             #18.30
        movl      %eax, %r14d                                   #18.30
        movd      %r9d, %xmm3                                   #20.40
        movl      %esi, %r9d                                    #19.29
        imull     %r10d, %r9d                                   #19.29
        movl      %ecx, %ebx                                    #18.30
        pshufd    $0, %xmm3, %xmm3                              #20.40
        movsd     %xmm13, (%rsp)                                #15.19
        movhpd    8(%rsp), %xmm15                               #15.19
        movdqa    L_2il0floatpacket.1(%rip), %xmm9              #21.3
        movl      %r10d, %r15d                                  #18.30
        movl      %r11d, %r13d                                  #18.30
        movd      %r9d, %xmm1                                   #19.29
        pshufd    $0, %xmm1, %xmm1                              #19.29
        movdqa    %xmm7, 48(%rsp)                               #18.30[spill]
        movdqa    %xmm8, 128(%rsp)                              #18.30[spill]
        movdqa    %xmm14, 80(%rsp)                              #18.30[spill]
        movdqa    %xmm1, 96(%rsp)                               #18.30[spill]
        movdqa    %xmm2, 112(%rsp)                              #18.30[spill]
        movdqa    %xmm3, 64(%rsp)                               #18.30[spill]
        movl      %esi, 16(%rsp)                                #18.30[spill]
        movl      %edi, 24(%rsp)                                #18.30[spill]
        movl      %r8d, 32(%rsp)                                #18.30[spill]
        movq      %rbp, 40(%rsp)                                #18.30[spill]
L_LCFI20:
        movq      %rdx, %rbp                                    #18.30
                                # LOE rbp ebx r12d r13d r14d r15d xmm4 xmm6 xmm9 xmm13 xmm15
L_B2.4:                         # Preds L_B2.4 L_B2.3
                                # Execution count [5.00e+00]
        movdqa    64(%rsp), %xmm1                               #20.40[spill]
        movdqa    %xmm9, %xmm0                                  #20.40
        movaps    %xmm6, (%rsp)                                 #[spill]
        addl      $4, %r14d                                     #16.2
        movaps    %xmm4, 144(%rsp)                              #[spill]
        call      ___svml_idiv4                                 #20.54
        movdqa    112(%rsp), %xmm1                              #20.54[spill]
        call      ___svml_irem4                                 #20.54
        movd      %xmm0, %rdi                                   #21.10
        psrldq    $8, %xmm0                                     #21.10
        movdqa    96(%rsp), %xmm1                               #19.29[spill]
        movd      %xmm0, %r9                                    #21.10
        movdqa    %xmm9, %xmm0                                  #19.29
        movslq    %edi, %rsi                                    #21.10
        movslq    %r9d, %r8                                     #21.10
        sarq      $32, %rdi                                     #21.10
        sarq      $32, %r9                                      #21.10
        movq      (%rbp,%rsi,8), %xmm12                         #21.10
        movq      (%rbp,%r8,8), %xmm11                          #21.10
        movhpd    (%rbp,%rdi,8), %xmm12                         #21.10
        movhpd    (%rbp,%r9,8), %xmm11                          #21.10
        call      ___svml_idiv4                                 #19.43
        movdqa    80(%rsp), %xmm1                               #19.43[spill]
        call      ___svml_irem4                                 #19.43
        movdqa    %xmm0, %xmm6                                  #21.10
        punpckldq %xmm0, %xmm6                                  #21.10
        movdqa    %xmm6, %xmm4                                  #21.10
        movdqa    L_2il0floatpacket.3(%rip), %xmm10             #21.10
        psrad     $31, %xmm4                                    #21.10
        pand      L_2il0floatpacket.2(%rip), %xmm4              #21.10
        pand      %xmm10, %xmm6                                 #21.10
        por       %xmm6, %xmm4                                  #21.10
        psllq     $3, %xmm4                                     #21.10
        psrldq    $8, %xmm0                                     #21.10
        paddq     %xmm4, %xmm12                                 #21.10
        punpckldq %xmm0, %xmm0                                  #21.10
        movd      %xmm12, %rsi                                  #21.10
        movdqa    %xmm0, %xmm2                                  #21.10
        psrad     $31, %xmm2                                    #21.10
        pand      %xmm10, %xmm0                                 #21.10
        pand      L_2il0floatpacket.2(%rip), %xmm2              #21.10
        punpckhqdq %xmm12, %xmm12                               #21.10
        por       %xmm0, %xmm2                                  #21.10
        movq      (%rsi), %xmm8                                 #21.10
        psllq     $3, %xmm2                                     #21.10
        movd      %xmm12, %rsi                                  #21.10
        movdqa    %xmm9, %xmm0                                  #18.17
        paddq     %xmm2, %xmm11                                 #21.10
        movhpd    (%rsi), %xmm8                                 #21.10
        movd      %xmm11, %rsi                                  #21.10
        punpckhqdq %xmm11, %xmm11                               #21.10
        movq      (%rsi), %xmm14                                #21.10
        movd      %xmm11, %rsi                                  #21.10
        movdqa    128(%rsp), %xmm11                             #18.17[spill]
        movdqa    %xmm11, %xmm1                                 #18.17
        movhpd    (%rsi), %xmm14                                #21.10
        call      ___svml_idiv4                                 #18.30
        movdqa    48(%rsp), %xmm1                               #18.30[spill]
        call      ___svml_irem4                                 #18.30
        movdqa    %xmm0, %xmm3                                  #21.10
        movdqa    %xmm11, %xmm1                                 #17.11
        punpckldq %xmm0, %xmm3                                  #21.10
        movdqa    %xmm3, %xmm5                                  #21.10
        pand      %xmm10, %xmm3                                 #21.10
        psrad     $31, %xmm5                                    #21.10
        pand      L_2il0floatpacket.2(%rip), %xmm5              #21.10
        por       %xmm3, %xmm5                                  #21.10
        psllq     $3, %xmm5                                     #21.10
        psrldq    $8, %xmm0                                     #21.10
        paddq     %xmm5, %xmm8                                  #21.10
        punpckldq %xmm0, %xmm0                                  #21.10
        movd      %xmm8, %rsi                                   #21.10
        movdqa    %xmm0, %xmm7                                  #21.10
        psrad     $31, %xmm7                                    #21.10
        pand      %xmm10, %xmm0                                 #21.10
        pand      L_2il0floatpacket.2(%rip), %xmm7              #21.10
        punpckhqdq %xmm8, %xmm8                                 #21.10
        por       %xmm0, %xmm7                                  #21.10
        movq      (%rsi), %xmm12                                #21.10
        psllq     $3, %xmm7                                     #21.10
        movd      %xmm8, %rsi                                   #21.10
        movdqa    %xmm9, %xmm0                                  #17.11
        paddq     %xmm7, %xmm14                                 #21.10
        movhpd    (%rsi), %xmm12                                #21.10
        movd      %xmm14, %rsi                                  #21.10
        punpckhqdq %xmm14, %xmm14                               #21.10
        movq      (%rsi), %xmm8                                 #21.10
        movd      %xmm14, %rsi                                  #21.10
        movhpd    (%rsi), %xmm8                                 #21.10
        call      ___svml_irem4                                 #17.11
        movdqa    %xmm0, %xmm6                                  #17.11
        cmpl      %ebx, %r14d                                   #16.2
        punpckldq %xmm0, %xmm6                                  #17.11
        movdqa    %xmm6, %xmm4                                  #17.11
        pand      %xmm10, %xmm6                                 #17.11
        psrad     $31, %xmm4                                    #17.11
        pand      L_2il0floatpacket.2(%rip), %xmm4              #17.11
        por       %xmm6, %xmm4                                  #17.11
        psllq     $4, %xmm4                                     #21.3
        paddq     %xmm4, %xmm12                                 #21.3
        movd      %xmm12, %rsi                                  #21.10
        psrldq    $8, %xmm0                                     #17.11
        punpckldq %xmm0, %xmm0                                  #17.11
        punpckhqdq %xmm12, %xmm12                               #21.10
        movdqa    %xmm0, %xmm1                                  #17.11
        movq      (%rsi), %rdi                                  #21.10
        psrad     $31, %xmm1                                    #17.11
        movq      8(%rsi), %rsi                                 #21.10
        pand      %xmm10, %xmm0                                 #17.11
        pand      L_2il0floatpacket.2(%rip), %xmm1              #17.11
        movd      %rdi, %xmm5                                   #21.10
        movd      %rsi, %xmm2                                   #21.10
        por       %xmm0, %xmm1                                  #17.11
        movd      %xmm12, %rsi                                  #21.10
        psllq     $4, %xmm1                                     #21.3
        paddq     %xmm1, %xmm8                                  #21.3
        movq      (%rsi), %rdi                                  #21.10
        movq      8(%rsi), %rsi                                 #21.10
        movd      %rdi, %xmm10                                  #21.10
        movd      %rsi, %xmm12                                  #21.10
        movd      %xmm8, %rsi                                   #21.10
        punpckhqdq %xmm8, %xmm8                                 #21.10
        unpcklpd  %xmm2, %xmm5                                  #21.10
        unpcklpd  %xmm12, %xmm10                                #21.10
        movq      (%rsi), %rdi                                  #21.10
        movq      8(%rsi), %rsi                                 #21.10
        movd      %rdi, %xmm11                                  #21.10
        movd      %rsi, %xmm3                                   #21.10
        movd      %xmm8, %rsi                                   #21.10
        unpcklpd  %xmm3, %xmm11                                 #21.10
        movaps    (%rsp), %xmm6                                 #21.3[spill]
        movaps    144(%rsp), %xmm4                              #21.3[spill]
        movq      (%rsi), %rdi                                  #21.10
        movq      8(%rsi), %rsi                                 #21.10
        movd      %rdi, %xmm14                                  #21.10
        movd      %rsi, %xmm8                                   #21.10
        unpcklpd  %xmm8, %xmm14                                 #21.10
        addpd     %xmm5, %xmm15                                 #21.3
        addpd     %xmm10, %xmm13                                #21.3
        addpd     %xmm11, %xmm6                                 #21.3
        addpd     %xmm14, %xmm4                                 #21.3
        paddd     L_2il0floatpacket.0(%rip), %xmm9              #21.3
        jb        L_B2.4        # Prob 82%                      #16.2
                                # LOE rbp ebx r12d r13d r14d r15d xmm4 xmm6 xmm9 xmm13 xmm15
L_B2.5:                         # Preds L_B2.4
                                # Execution count [9.00e-01]
        addpd     %xmm13, %xmm15                                #15.19
        addpd     %xmm6, %xmm15                                 #15.19
        addpd     %xmm4, %xmm15                                 #15.19
        movaps    %xmm15, (%rsp)                                #15.19
        movl      %ebx, %ecx                                    #
        movsd     (%rsp), %xmm13                                #15.19
        movq      %rbp, %rdx                                    #
        movl      16(%rsp), %esi                                #[spill]
        movl      %r13d, %r11d                                  #
        movl      24(%rsp), %edi                                #[spill]
        movl      %r15d, %r10d                                  #
        movl      32(%rsp), %r8d                                #[spill]
        movq      40(%rsp), %rbp                                #[spill]
L_LCFI21:
        movhpd    8(%rsp), %xmm13                               #15.19
                                # LOE rdx rbx rbp r13 r14 r15 ecx esi edi r8d r10d r11d r12d xmm13
L_B2.6:                         # Preds L_B2.5 L_B2.11
                                # Execution count [1.00e+00]
        cmpl      %r11d, %ecx                                   #16.2
        jae       L_B2.10       # Prob 9%                       #16.2
                                # LOE rdx rbx rbp r13 r14 r15 ecx esi edi r8d r10d r11d r12d xmm13
L_B2.7:                         # Preds L_B2.6
                                # Execution count [9.00e-01]
        movl      %esi, %r9d                                    #19.29
        movq      %rdx, %r15                                    #19.29
        imull     %r10d, %r12d                                  #20.40
        imull     %r10d, %r9d                                   #19.29
        .align    4
                                # LOE rbp r15 ecx esi edi r8d r9d r10d r11d r12d xmm13
L_B2.8:                         # Preds L_B2.8 L_B2.7
                                # Execution count [5.00e+00]
        movl      %ecx, %eax                                    #17.11
        cltd                                                    #17.11
        idivl     %r10d                                         #17.11
        movslq    %edx, %r13                                    #21.10
        cltd                                                    #18.30
        idivl     %esi                                          #18.30
        movl      %ecx, %eax                                    #19.29
        movslq    %edx, %rbx                                    #21.10
        cltd                                                    #19.29
        idivl     %r9d                                          #19.29
        shlq      $4, %r13                                      #21.10
        cltd                                                    #19.43
        idivl     %edi                                          #19.43
        movl      %ecx, %eax                                    #20.40
        incl      %ecx                                          #16.2
        movslq    %edx, %r14                                    #21.10
        cltd                                                    #20.40
        idivl     %r12d                                         #20.40
        cltd                                                    #20.54
        idivl     %r8d                                          #20.54
        cmpl      %r11d, %ecx                                   #16.2
        movslq    %edx, %rdx                                    #21.10
        movq      (%r15,%rdx,8), %rax                           #21.10
        movq      (%rax,%r14,8), %r14                           #21.10
        movq      (%r14,%rbx,8), %rax                           #21.10
        movsd     (%rax,%r13), %xmm1                            #21.10
        movhpd    8(%rax,%r13), %xmm1                           #21.10
        addpd     %xmm1, %xmm13                                 #21.3
        jb        L_B2.8        # Prob 82%                      #16.2
                                # LOE rbp r15 ecx esi edi r8d r9d r10d r11d r12d xmm13
L_B2.10:                        # Preds L_B2.8 L_B2.6 L_B2.1
                                # Execution count [1.00e+00]
        movaps    %xmm13, %xmm1                                 #23.9
        movaps    %xmm13, %xmm0                                 #23.9
        unpckhpd  %xmm13, %xmm1                                 #23.9
        addq      $160, %rsp                                    #23.9
L_LCFI22:
        popq      %rbx                                          #23.9
L_LCFI23:
        popq      %r15                                          #23.9
L_LCFI24:
        popq      %r14                                          #23.9
L_LCFI25:
        popq      %r13                                          #23.9
L_LCFI26:
        popq      %r12                                          #23.9
L_LCFI27:
        ret                                                     #23.9
L_LCFI28:
                                # LOE
L_B2.11:                        # Preds L_B2.2
                                # Execution count [9.00e-02]: Infreq
        xorl      %ecx, %ecx                                    #16.2
        jmp       L_B2.6        # Prob 100%                     #16.2
        .align    4
                                # LOE rdx rbx rbp r13 r14 r15 ecx esi edi r8d r10d r11d r12d xmm13
L_LCFI29:
# mark_end;
L..LN_Sum.1:
	.section	__DATA, __data
# -- End  _Sum
	.section	__TEXT, __text
L_2__routine_start__Contract_2:
# -- Begin  _Contract
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Contract
# --- Contract(complex128_t ****, const int *, complex128_t **)
_Contract:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
L_B3.1:                         # Preds L_B3.0
                                # Execution count [1.00e+00]
L_LCFI30:
L____tag_value__Contract.125:
L_L126:
                                                        #26.83
        pushq     %r12                                          #26.83
L_LCFI31:
        movq      %rdx, %rcx                                    #26.83
        movl      4(%rsi), %edx                                 #29.30
        movl      %edx, %eax                                    #29.30
        movl      12(%rsi), %r8d                                #29.52
        movl      %r8d, %r9d                                    #29.52
        movl      (%rsi), %r12d                                 #29.19
        movl      8(%rsi), %r11d                                #29.41
        imull     %r12d, %r9d                                   #29.52
        imull     %r11d, %eax                                   #29.30
        imull     %r9d, %eax                                    #29.41
        xorl      %r9d, %r9d                                    #30.2
        testl     %eax, %eax                                    #30.18
        jle       L_B3.5        # Prob 9%                       #30.18
                                # LOE rcx rbx rbp rdi r13 r14 r15 eax edx r8d r9d r11d r12d
L_B3.2:                         # Preds L_B3.1
                                # Execution count [9.00e-01]
        movl      %r11d, %esi                                   #34.29
        movl      %r11d, %r10d                                  #33.29
        imull     %edx, %esi                                    #34.29
        imull     %r8d, %r10d                                   #33.29
        imull     %r8d, %esi                                    #34.40
        movl      %eax, -24(%rsp)                               #34.40[spill]
        movl      %edx, -16(%rsp)                               #34.40[spill]
        movq      %r13, -32(%rsp)                               #34.40[spill]
        movq      %r14, -40(%rsp)                               #34.40[spill]
        movq      %r15, -48(%rsp)                               #34.40[spill]
        movq      %rbx, -56(%rsp)                               #34.40[spill]
        movq      %rbp, -64(%rsp)                               #34.40[spill]
L_LCFI32:
                                # LOE rcx rdi esi r8d r9d r10d r11d r12d
L_B3.3:                         # Preds L_B3.3 L_B3.2
                                # Execution count [5.00e+00]
        movl      %r9d, %eax                                    #33.29
        cltd                                                    #33.29
        idivl     %r10d                                         #33.29
        cltd                                                    #33.43
        idivl     -16(%rsp)                                     #33.43[spill]
        movl      %r9d, %eax                                    #34.40
        movslq    %edx, %rbp                                    #33.3
        cltd                                                    #34.40
        idivl     %esi                                          #34.40
        movq      %rbp, %r14                                    #35.3
        cltd                                                    #34.54
        idivl     %r12d                                         #34.54
        movl      %r9d, %eax                                    #31.11
        incl      %r9d                                          #30.2
        movslq    %edx, %rbx                                    #34.3
        cltd                                                    #31.11
        idivl     %r8d                                          #31.11
        movslq    %edx, %r13                                    #35.23
        cltd                                                    #32.30
        idivl     %r11d                                         #32.30
        movq      (%rcx,%rbx,8), %r15                           #35.3
        movq      (%rdi,%rbx,8), %rbx                           #35.23
        movslq    %edx, %rdx                                    #35.23
        shlq      $4, %r14                                      #35.3
        movq      (%rbx,%rbp,8), %rbp                           #35.23
        shlq      $4, %r13                                      #35.23
        cmpl      -24(%rsp), %r9d                               #30.2[spill]
        movq      (%rbp,%rdx,8), %rax                           #35.23
        movsd     (%r14,%r15), %xmm1                            #35.3
        movhpd    8(%r14,%r15), %xmm1                           #35.3
        movsd     (%rax,%r13), %xmm0                            #35.23
        movhpd    8(%rax,%r13), %xmm0                           #35.23
        addpd     %xmm0, %xmm1                                  #35.3
        movsd     %xmm1, (%r14,%r15)                            #35.3
        movhpd    %xmm1, 8(%r14,%r15)                           #35.3
        jb        L_B3.3        # Prob 82%                      #30.2
                                # LOE rcx rdi esi r8d r9d r10d r11d r12d
L_B3.4:                         # Preds L_B3.3
                                # Execution count [9.00e-01]
        movq      -32(%rsp), %r13                               #[spill]
L_LCFI33:
        movq      -40(%rsp), %r14                               #[spill]
L_LCFI34:
        movq      -48(%rsp), %r15                               #[spill]
L_LCFI35:
        movq      -56(%rsp), %rbx                               #[spill]
L_LCFI36:
        movq      -64(%rsp), %rbp                               #[spill]
L_LCFI37:
                                # LOE rbx rbp r13 r14 r15
L_B3.5:                         # Preds L_B3.4 L_B3.1
                                # Execution count [1.00e+00]
L_LCFI38:
        popq      %r12                                          #37.1
L_LCFI39:
        ret                                                     #37.1
        .align    4
                                # LOE
L_LCFI40:
# mark_end;
L..LN_Contract.2:
	.section	__DATA, __data
# -- End  _Contract
	.section	__TEXT, __text
L_2__routine_start__Offset_3:
# -- Begin  _Offset
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Offset
# --- Offset(complex128_t ****, const int *)
_Offset:
# parameter 1: %rdi
# parameter 2: %rsi
L_B4.1:                         # Preds L_B4.0
                                # Execution count [1.00e+00]
L_LCFI41:
L____tag_value__Offset.155:
L_L156:
                                                        #39.54
        subq      $104, %rsp                                    #39.54
L_LCFI42:
        xorl      %eax, %eax                                    #44.7
        movslq    (%rsi), %rcx                                  #44.18
        testq     %rcx, %rcx                                    #44.18
        jle       L_B4.19       # Prob 10%                      #44.18
                                # LOE rax rcx rbx rbp rsi rdi r12 r13 r14 r15
L_B4.2:                         # Preds L_B4.1
                                # Execution count [9.00e-01]
        movslq    4(%rsi), %rdx                                 #45.19
        movq      %r12, (%rsp)                                  #45.19[spill]
        movq      %r13, 8(%rsp)                                 #45.19[spill]
        movq      %r14, 16(%rsp)                                #45.19[spill]
        movq      %r15, 24(%rsp)                                #45.19[spill]
        movq      %rbx, 32(%rsp)                                #45.19[spill]
        movq      %rbp, 40(%rsp)                                #45.19[spill]
L_LCFI43:
                                # LOE rax rdx rcx rsi rdi
L_B4.3:                         # Preds L_B4.17 L_B4.2
                                # Execution count [5.00e+00]
        xorl      %r15d, %r15d                                  #45.8
        testq     %rdx, %rdx                                    #45.19
        jle       L_B4.17       # Prob 10%                      #45.19
                                # LOE rax rdx rcx rsi rdi r15
L_B4.4:                         # Preds L_B4.3
                                # Execution count [4.50e+00]
        movslq    8(%rsi), %r14                                 #46.20
        movq      %rcx, 48(%rsp)                                #46.20[spill]
                                # LOE rax rdx rsi rdi r14 r15
L_B4.5:                         # Preds L_B4.15 L_B4.4
                                # Execution count [2.50e+01]
        xorl      %r12d, %r12d                                  #46.9
        testq     %r14, %r14                                    #46.20
        jle       L_B4.15       # Prob 10%                      #46.20
                                # LOE rax rdx rsi rdi r12 r14 r15
L_B4.6:                         # Preds L_B4.5
                                # Execution count [2.25e+01]
        movslq    12(%rsi), %r13                                #47.21
        movq      %rdx, 64(%rsp)                                #47.21[spill]
        movq      %rsi, 56(%rsp)                                #47.21[spill]
                                # LOE rax rdi r12 r13 r14 r15
L_B4.7:                         # Preds L_B4.13 L_B4.6
                                # Execution count [1.25e+02]
        xorl      %ebp, %ebp                                    #47.10
        testq     %r13, %r13                                    #47.21
        jle       L_B4.13       # Prob 10%                      #47.21
                                # LOE rax rbp rdi r12 r13 r14 r15
L_B4.8:                         # Preds L_B4.7
                                # Execution count [1.12e+02]
        movq      (%rdi,%rax,8), %rdx                           #48.6
        movq      %rax, 80(%rsp)                                #48.6[spill]
        movq      %rdi, 72(%rsp)                                #48.6[spill]
        movq      (%rdx,%r15,8), %rcx                           #48.6
        movq      (%rcx,%r12,8), %rbx                           #48.6
                                # LOE rbx rbp r12 r13 r14 r15
L_B4.9:                         # Preds L_B4.25 L_B4.8
                                # Execution count [6.25e+02]
#       rand(void)
        call      _rand                                         #48.33
                                # LOE rbx rbp r12 r13 r14 r15 eax
L_B4.22:                        # Preds L_B4.9
                                # Execution count [6.25e+02]
        movl      %eax, %edi                                    #48.33
                                # LOE rbx rbp r12 r13 r14 r15 edi
L_B4.10:                        # Preds L_B4.22
                                # Execution count [6.25e+02]
        movl      $1374389535, %eax                             #48.25
        movl      %edi, %ecx                                    #48.25
        imull     %edi                                          #48.25
        sarl      $31, %ecx                                     #48.25
        xorps     %xmm1, %xmm1                                  #48.25
        sarl      $5, %edx                                      #48.25
        subl      %ecx, %edx                                    #48.25
        imull     $-100, %edx, %esi                             #48.25
        movsd     L_2il0floatpacket.4(%rip), %xmm0              #48.25
        addl      %esi, %edi                                    #48.25
        cvtsi2sd  %edi, %xmm1                                   #48.25
        call      _pow                                          #48.25
                                # LOE rbx rbp r12 r13 r14 r15 xmm0
L_B4.24:                        # Preds L_B4.10
                                # Execution count [6.25e+02]
        movsd     %xmm0, 88(%rsp)                               #48.25[spill]
#       rand(void)
        call      _rand                                         #48.61
                                # LOE rbx rbp r12 r13 r14 r15 eax
L_B4.23:                        # Preds L_B4.24
                                # Execution count [6.25e+02]
        movl      %eax, %edi                                    #48.61
                                # LOE rbx rbp r12 r13 r14 r15 edi
L_B4.11:                        # Preds L_B4.23
                                # Execution count [6.25e+02]
        movl      $1374389535, %eax                             #48.53
        movl      %edi, %ecx                                    #48.53
        imull     %edi                                          #48.53
        sarl      $31, %ecx                                     #48.53
        xorps     %xmm1, %xmm1                                  #48.53
        sarl      $5, %edx                                      #48.53
        subl      %ecx, %edx                                    #48.53
        imull     $-100, %edx, %esi                             #48.53
        movsd     L_2il0floatpacket.4(%rip), %xmm0              #48.53
        addl      %esi, %edi                                    #48.53
        cvtsi2sd  %edi, %xmm1                                   #48.53
        call      _pow                                          #48.53
                                # LOE rbx rbp r12 r13 r14 r15 xmm0
L_B4.25:                        # Preds L_B4.11
                                # Execution count [6.25e+02]
        movsd     88(%rsp), %xmm1                               #48.53[spill]
        incq      %rbp                                          #47.31
        movsd     (%rbx), %xmm2                                 #48.6
        movhpd    8(%rbx), %xmm2                                #48.6
        unpcklpd  %xmm0, %xmm1                                  #48.53
        addpd     %xmm1, %xmm2                                  #48.6
        movsd     %xmm2, (%rbx)                                 #48.6
        movhpd    %xmm2, 8(%rbx)                                #48.6
        addq      $16, %rbx                                     #47.31
        cmpq      %r13, %rbp                                    #47.21
        jl        L_B4.9        # Prob 82%                      #47.21
                                # LOE rbx rbp r12 r13 r14 r15
L_B4.12:                        # Preds L_B4.25
                                # Execution count [1.13e+02]
        movq      80(%rsp), %rax                                #[spill]
        movq      72(%rsp), %rdi                                #[spill]
                                # LOE rax rdi r12 r13 r14 r15
L_B4.13:                        # Preds L_B4.12 L_B4.7
                                # Execution count [1.25e+02]
        incq      %r12                                          #46.30
        cmpq      %r14, %r12                                    #46.20
        jl        L_B4.7        # Prob 82%                      #46.20
                                # LOE rax rdi r12 r13 r14 r15
L_B4.14:                        # Preds L_B4.13
                                # Execution count [2.25e+01]
        movq      64(%rsp), %rdx                                #[spill]
        movq      56(%rsp), %rsi                                #[spill]
                                # LOE rax rdx rsi rdi r14 r15
L_B4.15:                        # Preds L_B4.14 L_B4.5
                                # Execution count [2.50e+01]
        incq      %r15                                          #45.29
        cmpq      %rdx, %r15                                    #45.19
        jl        L_B4.5        # Prob 82%                      #45.19
                                # LOE rax rdx rsi rdi r14 r15
L_B4.16:                        # Preds L_B4.15
                                # Execution count [4.50e+00]
        .byte     15                                            #
        .byte     31                                            #
        .byte     0                                             #
        movq      48(%rsp), %rcx                                #[spill]
                                # LOE rax rdx rcx rsi rdi
L_B4.17:                        # Preds L_B4.16 L_B4.3
                                # Execution count [5.00e+00]
        incq      %rax                                          #44.28
        cmpq      %rcx, %rax                                    #44.18
        jl        L_B4.3        # Prob 82%                      #44.18
                                # LOE rax rdx rcx rsi rdi
L_B4.18:                        # Preds L_B4.17
                                # Execution count [9.00e-01]
        movq      (%rsp), %r12                                  #[spill]
L_LCFI44:
        movq      8(%rsp), %r13                                 #[spill]
L_LCFI45:
        movq      16(%rsp), %r14                                #[spill]
L_LCFI46:
        movq      24(%rsp), %r15                                #[spill]
L_LCFI47:
        movq      32(%rsp), %rbx                                #[spill]
L_LCFI48:
        movq      40(%rsp), %rbp                                #[spill]
L_LCFI49:
                                # LOE rbx rbp r12 r13 r14 r15
L_B4.19:                        # Preds L_B4.18 L_B4.1
                                # Execution count [1.00e+00]
        addq      $104, %rsp                                    #49.1
L_LCFI50:
        ret                                                     #49.1
        .align    4
                                # LOE
L_LCFI51:
# mark_end;
L..LN_Offset.3:
	.section	__DATA, __data
# -- End  _Offset
	.section	__TEXT, __const
	.align 4
L_2il0floatpacket.0:
	.long	0x00000004,0x00000004,0x00000004,0x00000004
	.align 4
L_2il0floatpacket.1:
	.long	0x00000000,0x00000001,0x00000002,0x00000003
	.align 4
L_2il0floatpacket.2:
	.long	0x00000000,0xffffffff,0x00000000,0xffffffff
	.align 4
L_2il0floatpacket.3:
	.long	0xffffffff,0x00000000,0xffffffff,0x00000000
	.cstring
	.align 2
	.align 2
L_2__STRING.0:
	.long	544044371
	.long	1629513327
	.long	1696623724
	.long	1701668204
	.long	544437358
	.long	1730486333
	.long	1763715872
	.long	778511648
	.word	10
	.literal8
	.align 3
	.align 3
L_2il0floatpacket.4:
	.long	0x00000000,0xbff00000
	.section	__DATA, __data
	.globl _main.eh
	.globl _Sum.eh
	.globl _Contract.eh
	.globl _Offset.eh
// -- Begin SEGMENT __eh_frame
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
__eh_frame_seg:
L.__eh_frame_seg:
EH_frame0:
L_fde_cie_0:
	.long 0x00000014
	.long 0x00000000
	.long 0x00527a01
	.long 0x01107801
	.long 0x08070c10
	.long 0x01900190
_main.eh:
	.long 0x0000010c
	.long 0x0000001c
	.quad L_LCFI1-_main.eh-0x8
	.set L_Qlab1,L_LCFI12-L_LCFI1
	.quad L_Qlab1
	.short 0x0400
	.set L_lab1,L_LCFI2-L_LCFI1
	.long L_lab1
	.short 0x100e
	.byte 0x04
	.set L_lab2,L_LCFI3-L_LCFI2
	.long L_lab2
	.long 0x8610060c
	.short 0x0402
	.set L_lab3,L_LCFI4-L_LCFI3
	.long L_lab3
	.long 0x380e0310
	.long 0xff800d1c
	.long 0x0d1affff
	.long 0xffffffd8
	.long 0x0e0c1022
	.long 0x800d1c38
	.long 0x1affffff
	.long 0xfffff80d
	.long 0x0d1022ff
	.long 0x0d1c380e
	.long 0xffffff80
	.long 0xfff00d1a
	.long 0x1022ffff
	.long 0x1c380e0e
	.long 0xffff800d
	.long 0xe80d1aff
	.long 0x22ffffff
	.long 0x380e0f10
	.long 0xff800d1c
	.long 0x0d1affff
	.long 0xffffffe0
	.short 0x0422
	.set L_lab4,L_LCFI5-L_LCFI4
	.long L_lab4
	.short 0x04c3
	.set L_lab5,L_LCFI6-L_LCFI5
	.long L_lab5
	.short 0x04cf
	.set L_lab6,L_LCFI7-L_LCFI6
	.long L_lab6
	.short 0x04ce
	.set L_lab7,L_LCFI8-L_LCFI7
	.long L_lab7
	.short 0x04cd
	.set L_lab8,L_LCFI9-L_LCFI8
	.long L_lab8
	.short 0x04cc
	.set L_lab9,L_LCFI10-L_LCFI9
	.long L_lab9
	.long 0xc608070c
	.byte 0x04
	.set L_lab10,L_LCFI11-L_LCFI10
	.long L_lab10
	.long 0x1010060c
	.long 0x1c380e03
	.long 0xffff800d
	.long 0xd80d1aff
	.long 0x22ffffff
	.long 0x0c100286
	.long 0x0d1c380e
	.long 0xffffff80
	.long 0xfff80d1a
	.long 0x1022ffff
	.long 0x1c380e0d
	.long 0xffff800d
	.long 0xf00d1aff
	.long 0x22ffffff
	.long 0x380e0e10
	.long 0xff800d1c
	.long 0x0d1affff
	.long 0xffffffe8
	.long 0x0e0f1022
	.long 0x800d1c38
	.long 0x1affffff
	.long 0xffffe00d
	.long 0x000022ff
	.long 0x00000000
_Sum.eh:
	.long 0x0000009c
	.long 0x0000012c
	.quad L_LCFI13-_Sum.eh-0x8
	.set L_Qlab2,L_LCFI29-L_LCFI13
	.quad L_Qlab2
	.short 0x0400
	.set L_lab11,L_LCFI14-L_LCFI13
	.long L_lab11
	.long 0x028c100e
	.byte 0x04
	.set L_lab12,L_LCFI15-L_LCFI14
	.long L_lab12
	.long 0x038d180e
	.byte 0x04
	.set L_lab13,L_LCFI16-L_LCFI15
	.long L_lab13
	.long 0x048e200e
	.byte 0x04
	.set L_lab14,L_LCFI17-L_LCFI16
	.long L_lab14
	.long 0x058f280e
	.byte 0x04
	.set L_lab15,L_LCFI18-L_LCFI17
	.long L_lab15
	.long 0x0683300e
	.byte 0x04
	.set L_lab16,L_LCFI19-L_LCFI18
	.long L_lab16
	.long 0x0401d00e
	.set L_lab17,L_LCFI20-L_LCFI19
	.long L_lab17
	.short 0x1586
	.byte 0x04
	.set L_lab18,L_LCFI21-L_LCFI20
	.long L_lab18
	.short 0x04c6
	.set L_lab19,L_LCFI22-L_LCFI21
	.long L_lab19
	.long 0x04c3300e
	.set L_lab20,L_LCFI23-L_LCFI22
	.long L_lab20
	.long 0x04cf280e
	.set L_lab21,L_LCFI24-L_LCFI23
	.long L_lab21
	.long 0x04ce200e
	.set L_lab22,L_LCFI25-L_LCFI24
	.long L_lab22
	.long 0x04cd180e
	.set L_lab23,L_LCFI26-L_LCFI25
	.long L_lab23
	.long 0x04cc100e
	.set L_lab24,L_LCFI27-L_LCFI26
	.long L_lab24
	.short 0x080e
	.byte 0x04
	.set L_lab25,L_LCFI28-L_LCFI27
	.long L_lab25
	.long 0x8301d00e
	.long 0x8d028c06
	.long 0x8f048e03
	.long 0x00000005
	.byte 0x00
_Contract.eh:
	.long 0x0000005c
	.long 0x000001cc
	.quad L_LCFI30-_Contract.eh-0x8
	.set L_Qlab3,L_LCFI40-L_LCFI30
	.quad L_Qlab3
	.short 0x0400
	.set L_lab26,L_LCFI31-L_LCFI30
	.long L_lab26
	.long 0x028c100e
	.byte 0x04
	.set L_lab27,L_LCFI32-L_LCFI31
	.long L_lab27
	.long 0x0a860983
	.long 0x078e068d
	.short 0x088f
	.byte 0x04
	.set L_lab28,L_LCFI33-L_LCFI32
	.long L_lab28
	.short 0x04cd
	.set L_lab29,L_LCFI34-L_LCFI33
	.long L_lab29
	.short 0x04ce
	.set L_lab30,L_LCFI35-L_LCFI34
	.long L_lab30
	.short 0x04cf
	.set L_lab31,L_LCFI36-L_LCFI35
	.long L_lab31
	.short 0x04c3
	.set L_lab32,L_LCFI37-L_LCFI36
	.long L_lab32
	.short 0x04c6
	.set L_lab33,L_LCFI38-L_LCFI37
	.long L_lab33
	.short 0x04cc
	.set L_lab34,L_LCFI39-L_LCFI38
	.long L_lab34
	.long 0x0000080e
	.short 0x0000
_Offset.eh:
	.long 0x0000005c
	.long 0x0000022c
	.quad L_LCFI41-_Offset.eh-0x8
	.set L_Qlab4,L_LCFI51-L_LCFI41
	.quad L_Qlab4
	.short 0x0400
	.set L_lab35,L_LCFI42-L_LCFI41
	.long L_lab35
	.short 0x700e
	.byte 0x04
	.set L_lab36,L_LCFI43-L_LCFI42
	.long L_lab36
	.long 0x09860a83
	.long 0x0d8d0e8c
	.long 0x0b8f0c8e
	.byte 0x04
	.set L_lab37,L_LCFI44-L_LCFI43
	.long L_lab37
	.short 0x04cc
	.set L_lab38,L_LCFI45-L_LCFI44
	.long L_lab38
	.short 0x04cd
	.set L_lab39,L_LCFI46-L_LCFI45
	.long L_lab39
	.short 0x04ce
	.set L_lab40,L_LCFI47-L_LCFI46
	.long L_lab40
	.short 0x04cf
	.set L_lab41,L_LCFI48-L_LCFI47
	.long L_lab41
	.short 0x04c3
	.set L_lab42,L_LCFI49-L_LCFI48
	.long L_lab42
	.short 0x04c6
	.set L_lab43,L_LCFI50-L_LCFI49
	.long L_lab43
	.long 0x0000080e
	.short 0x0000
# End
	.subsections_via_symbols
