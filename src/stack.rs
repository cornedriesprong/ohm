pub(crate) struct Stack<T = f32> {
    data: [T; 64],
    sp: usize,
}

impl<T: Copy + Default> Stack<T> {
    pub(crate) fn new() -> Self {
        Self {
            data: [T::default(); 64],
            sp: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.sp = 0;
    }

    pub(crate) fn push(&mut self, v: T) {
        self.data[self.sp] = v;
        self.sp += 1;
    }

    pub(crate) fn pop(&mut self) -> T {
        self.sp -= 1;
        self.data[self.sp]
    }

    pub(crate) fn top(&self) -> T {
        self.data[self.sp - 1]
    }

    pub(crate) fn top_mut(&mut self) -> &mut T {
        &mut self.data[self.sp - 1]
    }

    pub(crate) fn peek(&self, offset: usize) -> &T {
        &self.data[self.sp - 1 - offset]
    }
}
