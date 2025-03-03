const clamp = (num, min, max) => Math.min(Math.max(num, min), max)

const CellEvent = Object.freeze({
    CURSOR_IN: Symbol("CURSOR_IN"),
    CURSOR_OUT: Symbol("CURSOR_OUT"),
    DO_PAINT: Symbol("DO_PAINT")
});

const Grid = function (container) {
    this.Height = 0;
    this.Width = 0;
    this.Container = container;
    this.Cells = [];
    this.Display = [];
}

Grid.prototype = {
    Create(height, width) {
        this.Height = height;
        this.Width = width;
        let index = 0;
        for (let y = 0; y < height; y++) {
            let row = [];
            for (let x = 0; x < width; x++) {

                var activeCell = new ActiveCell(this, index++, x, y);
                row.push(activeCell);
                this.Cells.push(activeCell);

            }
            this.Display.push(row);
        }
    }

}

const ActiveCell = function (grid, index, x, y) {
    this.Grid = grid;
    this.Index = index;
    this.X = x;
    this.Y = y;
    this.brushRadius = 3;
    this.Value = 0;
    var element = document.createElement('div');
    element.draggable = false;
    element.className = 'cell';
    grid.Container.appendChild(element);
    this.Cell = element;

    element.addEventListener('mouseenter', (event) => {

        this.ApplyBrush({ event: event, action: CellEvent.CURSOR_IN });
    });

    element.addEventListener('mouseleave', (event) => {

        this.ApplyBrush({ event: event, action: CellEvent.CURSOR_OUT });
    });

    element.addEventListener('mouseup', (event) => {
        this.ApplyBrush({ event: event, action: CellEvent.DO_PAINT });
    });
}

ActiveCell.prototype = {
    Clear() {

    },
    ApplyBrush(ev) {
        let minX = 0;
        let minY = 0;
        let maxX = this.Grid.Width - 1;
        let maxY = this.Grid.Height - 1;

        if (ev.event.buttons == 1) {
            ev.action = CellEvent.DO_PAINT;
        }

        for (let y = clamp(this.Y - this.brushRadius, minY, maxY); y <= clamp(this.Y + this.brushRadius, minY, maxY); y++) {
            for (let x = clamp(this.X - this.brushRadius, minX, maxX); x <= clamp(this.X + this.brushRadius, minX, maxX); x++) {
                const distance = Math.sqrt(Math.pow(this.Y - y, 2) + Math.pow(this.X - x, 2));
                if (distance <= this.brushRadius) {
                    const opacity = (1.0 - (distance / this.brushRadius));

                    var currentCell = this.Grid.Display[y][x];



                    switch (ev.action) {
                        case CellEvent.CURSOR_IN:
                            if (currentCell.Value != 0) break;
                            currentCell.Cell.style.backgroundColor = `rgba(255, 204, 0, ${opacity})`;
                            break;
                        case CellEvent.CURSOR_OUT:
                            if (currentCell.Value != 0) break;
                            currentCell.Cell.style.backgroundColor = '';
                            break;
                        case CellEvent.DO_PAINT:
                            let val = opacity * 255;
                            if (currentCell.Value > val) break;
                            currentCell.Cell.style.backgroundColor = `rgba(0, 0, 0, ${opacity.toFixed(2)})`;
                            currentCell.Value = val;
                            break;
                        default:
                    }
                }
            }
        }
    }
}

const OutputGrid = function (container) {
    this.Container = container;
    this.Cells = [];
}

OutputGrid.prototype = {
    Create(size) {
        for (let i = 0; i < size; i++) {
            var displayCell = new ResultDisplay(this.Container, i);
            this.Cells.push(displayCell);
        }
    }
}

const ResultDisplay = function (container, index) {
    this.Container = container;
    this.Index = index;

    const template = document.getElementById('result');

    const clone = template.content.cloneNode(true);

    this.ProgressBar = clone.querySelector('.progress-bar'); 
    this.Label = clone.querySelector('.badge');

    this.ProgressBar.setAttribute('aria-label', index);
    this.Label.innerText = index;

    this.Container.appendChild(clone);
}

ResultDisplay.prototype = {

    SetValue(value) {
        this.ProgressBar.setAttribute('style', 'width:' + value + '%');
    }

}

